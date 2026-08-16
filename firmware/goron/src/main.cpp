// goron フェイス - Core2 の LCD に Cozmo 風の目を表示する
//
// References:
// https://docs.m5stack.com/en/core/core2
// https://github.com/m5stack/M5Unified
// https://github.com/lovyan03/LovyanGFX
//
// 黒背景に青い角丸四角の目を2つ描き、まばたき・視線移動を自律的に再生する。
// 全画面スプライト(M5Canvas)に描いてから一括転送することでちらつきを防ぐ。
//
// 移植元 (ai_robotics/firmware/stackchan) は上下まぶたで目を削って12種類の表情を
// 作っていたが、ここでは normal（まばたき・視線のゆらぎ・呼吸の上下動だけ）に絞る。
//
// 脚: DYNAMIXEL XL330-M288-T ×2（Extended Position Control Mode）
//   PORT.A に TTL インターフェース基板を介して接続。
//   基板が送受信を自動で切り替えるため DIR ピンの制御は不要。
//   起動時にクランクを1周回してから 0度 に戻し、配線・通信と原点を同時に確認する。

#include <M5Unified.h>
#include <Dynamixel2Arduino.h>
#include <math.h>

// --- 目のデザインパラメータ（見た目を調整する箇所） ---

static const int SCREEN_W = 320;
static const int SCREEN_H = 240;
static const int EYE_W    = 78;   // 目の幅
static const int EYE_H    = 96;   // 目の高さ（開いた状態）
static const int EYE_R    = 24;   // 角丸半径
static const int EYE_GAP  = 44;   // 左右の目の隙間
static const int EYE_CY   = 90;   // 目の中心Y（顔として見えるよう画面中央より上に置く）

static const uint16_t COLOR_EYE = 0x05FF;  // 明るいシアン (#00BFFF 相当)
static const uint16_t COLOR_BG  = TFT_BLACK;

// --- アニメーションパラメータ ---

static const uint32_t FRAME_MS     = 33;   // 約30fps
static const uint32_t BLINK_MS     = 150;  // まばたき1回にかける時間
static const uint32_t BLINK_MIN_MS = 2000; // まばたきの間隔（ランダム）
static const uint32_t BLINK_MAX_MS = 6000;
static const float    BLINK_OPEN   = 0.05f; // 閉じきったときの開き具合

static const uint32_t SACCADE_MS     = 120;  // 視線移動にかける時間
static const uint32_t SACCADE_MIN_MS = 1000; // 視線移動の間隔（ランダム）
static const uint32_t SACCADE_MAX_MS = 4000;
static const int      GAZE_RANGE_X   = 10;   // 視線オフセットの振れ幅
static const int      GAZE_RANGE_Y   = 8;

static const float    DRIFT_PX        = 2.0f;   // 呼吸のような緩やかな上下動
static const uint32_t DRIFT_PERIOD_MS = 4000;

// --- 脚（DYNAMIXEL XL330-M288-T）---

// PORT.A（G32/G33）。基板のサンプルは RX=32/TX=33 だが、実機はこの向きでしか応答しない
static const int      DXL_RX_PIN   = 33;
static const int      DXL_TX_PIN   = 32;
static const float    DXL_PROTOCOL = 2.0f;
static const uint32_t DXL_BAUD     = 1000000;
static const uint8_t  DXL_IDS[]    = {1, 2};  // 実機をスキャンして確認した ID
static const uint8_t  DXL_COUNT    = sizeof(DXL_IDS) / sizeof(DXL_IDS[0]);
static const uint32_t DXL_SETTLE_MS  = 200;
static const uint8_t  DXL_PING_RETRY = 3;

// 名前は ControlTableItem の列挙子と衝突するので DXL_ を付ける
static const uint16_t DXL_PROFILE_VEL = 100;  // ×0.229 rpm ≒ 23 rpm。机の上で危なくない速さ
static const uint16_t DXL_PROFILE_ACC = 20;   // ×214.577 rev/min^2
static const float    HOME_DEG         = 0.0f; // 原点（エンコーダ絶対値の0度）
static const float    REACH_TOLERANCE  = 3.0f; // 到達とみなす角度差
static const uint32_t REACH_TIMEOUT_MS = 8000; // 引っかかったときに止まらないための上限
static const uint32_t REACH_POLL_MS    = 20;   // 到達判定の問い合わせ間隔
static const uint32_t HOME_SHOW_MS     = 1000; // 起動メッセージを読む時間

// --- 状態 ---

static M5Canvas canvas(&M5.Display);

static uint32_t blinkAt  = 0;  // 次のまばたき開始時刻
static uint32_t blinkEnd = 0;  // まばたき終了時刻（過ぎていれば目は開いている）

static int      gazeX = 0, gazeY = 0;          // 現在の視線オフセット
static int      gazeFromX = 0, gazeFromY = 0;  // サッケードの始点
static int      gazeToX = 0, gazeToY = 0;      // サッケードの終点
static uint32_t gazeStart = 0;
static uint32_t gazeAt    = 0;  // 次のサッケード開始時刻

// --- 脚の原点出し ---

static Dynamixel2Arduino dxl;
using namespace ControlTableItem;

// 全モータが目標角に入るまで待つ。引っかかったまま戻らないようタイムアウトで打ち切る
static void waitReached(const uint8_t* ids, uint8_t count, const float* goals) {
    uint32_t start = millis();
    for (;;) {
        bool done = true;
        for (uint8_t i = 0; i < count; i++) {
            float present = dxl.getPresentPosition(ids[i], UNIT_DEGREE);
            if (fabsf(goals[i] - present) > REACH_TOLERANCE) done = false;
        }
        if (done) return;
        if (millis() - start > REACH_TIMEOUT_MS) {
            Serial.println("move timeout");
            return;
        }
        delay(REACH_POLL_MS);
    }
}

static void moveTo(const uint8_t* ids, uint8_t count, const float* goals) {
    for (uint8_t i = 0; i < count; i++) dxl.setGoalPosition(ids[i], goals[i], UNIT_DEGREE);
    waitReached(ids, count, goals);
}

// 起動時にクランクを1周させてから原点(0度)へ戻す。配線・通信の確認と原点出しを兼ねる
static void homeLegs() {
    Serial2.begin(DXL_BAUD, SERIAL_8N1, DXL_RX_PIN, DXL_TX_PIN);
    dxl = Dynamixel2Arduino(Serial2);
    dxl.begin(DXL_BAUD);
    dxl.setPortProtocolVersion(DXL_PROTOCOL);
    delay(DXL_SETTLE_MS);  // 開いた直後は最初のパケットが落ちることがある

    uint8_t ids[DXL_COUNT];
    uint8_t found = 0;
    for (uint8_t i = 0; i < DXL_COUNT; i++) {
        // 取りこぼしでモータを見失わないよう数回試す
        for (uint8_t retry = 0; retry < DXL_PING_RETRY; retry++) {
            if (dxl.ping(DXL_IDS[i])) { ids[found++] = DXL_IDS[i]; break; }
        }
    }
    if (found == 0) {
        Serial.println("No DYNAMIXEL found");
        M5.Display.println("No servos");
        return;
    }
    for (uint8_t i = 0; i < found; i++) {
        Serial.printf("DYNAMIXEL id=%u model=%u\n", ids[i], dxl.getModelNumber(ids[i]));
    }
    M5.Display.printf("Servos: %u\n", found);

    for (uint8_t i = 0; i < found; i++) {
        dxl.torqueOff(ids[i]);
        // 360°連続回転と絶対角フィードバックを両立できるのはこのモードだけ
        dxl.setOperatingMode(ids[i], OP_EXTENDED_POSITION);
        dxl.writeControlTableItem(PROFILE_VELOCITY, ids[i], DXL_PROFILE_VEL);
        dxl.writeControlTableItem(PROFILE_ACCELERATION, ids[i], DXL_PROFILE_ACC);
        dxl.torqueOn(ids[i]);
    }

    float goals[DXL_COUNT];
    for (uint8_t i = 0; i < found; i++) {
        goals[i] = dxl.getPresentPosition(ids[i], UNIT_DEGREE) + 360.0f;
    }
    M5.Display.println("Spin...");
    moveTo(ids, found, goals);

    for (uint8_t i = 0; i < found; i++) goals[i] = HOME_DEG;
    M5.Display.println("Home 0deg");
    moveTo(ids, found, goals);

    for (uint8_t i = 0; i < found; i++) {
        Serial.printf("id=%u home at %.1f deg\n", ids[i],
                      dxl.getPresentPosition(ids[i], UNIT_DEGREE));
    }
    // トルクは入れたまま。0度を保持して顔の再生に移る
}

// --- アニメーション ---

// 目の開き具合(0.0〜1.0)を返す。まばたきの開始判定もここで行う
static float eyeOpen(uint32_t now) {
    if (now >= blinkAt) {
        blinkEnd = now + BLINK_MS;
        blinkAt  = blinkEnd + random(BLINK_MIN_MS, BLINK_MAX_MS);
    }
    if (now >= blinkEnd) return 1.0f;
    // 閉じ→開きを1本の三角波で表す（p=0.5 で閉じきる）
    float p = 1.0f - (float)(blinkEnd - now) / BLINK_MS;
    return BLINK_OPEN + (1.0f - BLINK_OPEN) * fabsf(p * 2.0f - 1.0f);
}

// 視線のゆらぎを更新する。サッケードの開始判定もここで行う
static void updateGaze(uint32_t now) {
    if (now >= gazeAt) {
        gazeFromX = gazeX;
        gazeFromY = gazeY;
        gazeToX   = random(-GAZE_RANGE_X, GAZE_RANGE_X + 1);
        gazeToY   = random(-GAZE_RANGE_Y, GAZE_RANGE_Y + 1);
        gazeStart = now;
        gazeAt    = now + SACCADE_MS + random(SACCADE_MIN_MS, SACCADE_MAX_MS);
    }
    uint32_t t = now - gazeStart;
    float p = (t >= SACCADE_MS) ? 1.0f : (float)t / SACCADE_MS;
    p = p * p * (3.0f - 2.0f * p);  // smoothstep で動き出しと止まりを滑らかにする
    gazeX = gazeFromX + (int)((gazeToX - gazeFromX) * p);
    gazeY = gazeFromY + (int)((gazeToY - gazeFromY) * p);
}

// --- 描画 ---

// 1つの目を描く。open はまばたきによる縦の潰れ具合
static void drawEye(int cx, int cy, float open) {
    int h = max(4, (int)(EYE_H * open));
    int r = min(EYE_R, min(EYE_W, h) / 2);
    canvas.fillSmoothRoundRect(cx - EYE_W / 2, cy - h / 2, EYE_W, h, r, COLOR_EYE);
}

static void drawFace(uint32_t now) {
    float open  = eyeOpen(now);
    int   drift = (int)(DRIFT_PX * sinf(now * (2.0f * (float)M_PI / DRIFT_PERIOD_MS)));
    int   half  = (EYE_GAP + EYE_W) / 2;  // 画面中央から各目の中心までの距離

    updateGaze(now);
    int cx = SCREEN_W / 2 + gazeX;
    int cy = EYE_CY + gazeY + drift;

    canvas.fillSprite(COLOR_BG);
    drawEye(cx - half, cy, open);
    drawEye(cx + half, cy, open);
    canvas.pushSprite(0, 0);
}

// --- setup / loop ---

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    M5.Display.fillScreen(COLOR_BG);
    M5.Display.setTextSize(2);

    Serial.begin(115200);
    while (!Serial && millis() < 3000);  // 起動直後のログを取りこぼさない（最大3秒待つ）

    homeLegs();
    delay(HOME_SHOW_MS);
    M5.Display.fillScreen(COLOR_BG);

    canvas.setColorDepth(16);
    // PSRAM は書き込み帯域が狭くフレームレートに響くため、内蔵 SRAM を優先して確保する
    canvas.setPsram(false);
    if (!canvas.createSprite(SCREEN_W, SCREEN_H)) {
        canvas.setPsram(true);
        if (!canvas.createSprite(SCREEN_W, SCREEN_H)) {
            Serial.println("sprite alloc failed");
            return;
        }
        Serial.println("sprite: psram");
    } else {
        Serial.println("sprite: internal sram");
    }

    randomSeed(esp_random());
    uint32_t now = millis();
    blinkAt = now + random(BLINK_MIN_MS, BLINK_MAX_MS);
    gazeAt  = now + random(SACCADE_MIN_MS, SACCADE_MAX_MS);
}

void loop() {
    M5.update();

    static uint32_t nextFrame = 0;
    uint32_t now = millis();
    if (now < nextFrame) return;
    nextFrame = now + FRAME_MS;

    drawFace(now);
}
