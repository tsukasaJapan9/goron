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
#include <Preferences.h>
#include <math.h>

#include "imu_map.h"   // scripts/make_imu_header.py が hardware/imu_map.json から生成
#include GORON_POLICY  // scripts/export_policy.py が学習結果から生成

// --- 目のデザインパラメータ（見た目を調整する箇所） ---

static const int SCREEN_W = 320;
static const int SCREEN_H = 240;
static const int EYE_W    = 78;   // 目の幅
static const int EYE_H    = 96;   // 目の高さ（開いた状態）
static const int EYE_R    = 24;   // 角丸半径
static const int EYE_GAP  = 44;   // 左右の目の隙間
static const int EYE_CY   = 90;   // 目の中心Y（顔として見えるよう画面中央より上に置く）

static const uint16_t COLOR_EYE   = 0x05FF;  // 明るいシアン (#00BFFF 相当)
static const uint16_t COLOR_BG    = TFT_BLACK;
static const uint16_t COLOR_LABEL = 0x4208;  // 状態表示。目より目立たない暗いグレー
static const int      STATUS_MARGIN = 8;     // 状態表示と画面下端の間隔

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
// 実機をスキャンして確認した ID。並び順が左右を決める（先頭が左脚）。
// 観測 [sin(qL), sin(qR), cos(qL), cos(qR)] とテレメトリの並びがこれに従う。
//
// ここで言う「左」はモデルの +Y 側であって、見た目の左右ではない。モデルの +X は
// 画面と反対を向く（m5stack.stl は CAD Z -37.5..-21、CAD Z が sim の X）ので、
// 画面のある側を前だと思って数えると左右が入れ替わる。scripts/mirror で実機と
// 見比べて確定した並びがこれ。
static const uint8_t  DXL_IDS[]    = {1, 2};
static const uint8_t  DXL_COUNT    = sizeof(DXL_IDS) / sizeof(DXL_IDS[0]);
static const uint32_t DXL_SETTLE_MS  = 200;
static const uint8_t  DXL_PING_RETRY = 3;

// 名前は ControlTableItem の列挙子と衝突するので DXL_ を付ける
static const uint16_t DXL_PROFILE_ACC = 20;   // ×214.577 rev/min^2。急発進させない
static const uint16_t DXL_PROFILE_VEL = 100;  // ×0.229 rpm ≒ 23 rpm。ホームへ戻るときの速さ
static const float    HOME_DEG        = 90.0f;  // B ボタンで入れるクランク角（原点からの角度）

// C ボタンで較正するときに作る姿勢の、クランク角。
// 「胴体を床に伏せ、両脚の先端が床に触れる」姿勢。目分量の CAD 姿勢と違って
// 機械的に再現できるので原点の基準に向く。RobotParams.asbuilt() の寸法から
// 計算した値（ヒンジ高さ 37.4 mm、脚先半径 73.7 mm）。もう一方の交点 62.1 度は
// 腕の途中が当たるだけで先端は浮くため、基準にはならない。寸法を変えたら再計算する。
static const float    CALIB_POSE_DEG  = 279.6f;
static const char*    NVS_NAMESPACE   = "goron"; // 較正した原点の保存先

// --- 方策の実行 ---
//
// 観測・行動の規約はエクスポートしたヘッダの冒頭に書いてある。ここで守るべきは:
//   * 観測はシミュレータのボディ座標系。IMU は imu_map.h で回してから使う
//   * クランク角は較正した原点から。方策ごとの GORON_CRANK_ZERO_DEG を引く
//   * 行動は Extended Position 目標角の「増分」。実測角から GORON_MAX_LEAD 以内に
//     制限する（アンチワインドアップ。脚が引っかかったとき目標が無限に先走るのを防ぐ）
//   * プロファイル(台形加減速)は切る。入れると台形生成器の設定を追うことになり、
//     サーボ同定もこの条件で行っている
static const float GORON_NO_LOAD_SPEED = 10.79f;  // rad/s、観測の正規化に使う
static const float GORON_STALL_TORQUE  = 0.52f;   // N.m、同上
static const float DXL_SAT_CURRENT_MA  = 860.0f;  // 全開PWMで流れる電流の実測値
static const uint32_t POLICY_PERIOD_US = 1000000UL / GORON_CONTROL_HZ;

// --- テレメトリ（PC 側の scripts/mirror.py がこれを読んで MuJoCo に流す）---
static const uint32_t TELEM_MS = 50;  // 20Hz。目視で追う用途にはこれで足りる
// Aボタンで回すときの回転数（机から落とさない程度）。負 = 逆転
static const int      SPIN_RPM        = -20;
static const uint32_t BOOT_SHOW_MS    = 1000; // 起動メッセージを読む時間

// --- 状態 ---

static M5Canvas canvas(&M5.Display);

static uint32_t blinkAt  = 0;  // 次のまばたき開始時刻
static uint32_t blinkEnd = 0;  // まばたき終了時刻（過ぎていれば目は開いている）

static int      gazeX = 0, gazeY = 0;          // 現在の視線オフセット
static int      gazeFromX = 0, gazeFromY = 0;  // サッケードの始点
static int      gazeToX = 0, gazeToY = 0;      // サッケードの終点
static uint32_t gazeStart = 0;
static uint32_t gazeAt    = 0;  // 次のサッケード開始時刻

// --- 脚 ---

static Dynamixel2Arduino dxl;
using namespace ControlTableItem;

static uint8_t legIds[DXL_COUNT];
static uint8_t legCount = 0;
static bool    spinning = false;
static uint8_t legMode  = 0xFF;  // 現在の動作モード。0xFF = 未設定（最初は必ず書き込む）

// 制御ループは Core1 の専用タスクで回す。Core0 の顔の描画は PSRAM から
// 320x240 を転送するのに数十ms かかり、同じループに置くと 20ms の周期を守れない
static volatile bool policyRunning = false;
// 制御ループが読んだ値の写し。方策の実行中、テレメトリはこれを送る。
// テレメトリが自分でサーボを読むと Core0 と Core1 が同じ TTL バスを取り合って
// 通信が壊れるので、読むのは制御ループの1箇所だけにする
static volatile float lastRaw[DXL_COUNT], lastCrank[DXL_COUNT], lastImu[6];
static volatile uint32_t policyHz = 0;   // 実測の制御周波数
// 空回し。観測と方策の計算はするが目標角を送らない。IMU の軸・クランク角・
// 正規化・制御周期を、脚を動かさずに検証するため
static bool  policyDryRun = false;
static float policyTarget[DXL_COUNT];   // Extended Position の目標角[deg]（多回転のまま）
static float policyAction[GORON_ACT];   // 表示用
static uint32_t policyLateUs = 0;       // 周期に間に合わなかった最大の遅れ

static Preferences prefs;
static float legZero[DXL_COUNT] = {0.0f, 0.0f};  // CAD組立姿勢のサーボ角[deg]
static bool  calibrating = false;

// 角度差を (-180, 180] に畳む。どちら回りが近いかの判断に使う
static float wrapDeg(float d) {
    d = fmodf(d + 180.0f, 360.0f);
    if (d < 0.0f) d += 360.0f;
    return d - 180.0f;
}

// 較正した原点から見たクランク角 [0, 360)。
// XL330 は絶対エンコーダなので、再起動して多回転ぶんが失われても mod 360 で復元できる
static float crankDeg(uint8_t i, float present) {
    float d = fmodf(present - legZero[i], 360.0f);
    return (d < 0.0f) ? d + 360.0f : d;
}

static void loadZeros() {
    // 書き込み可で開く。読み出し専用だと初回（名前空間が無い）にエラーを吐く
    prefs.begin(NVS_NAMESPACE, false);
    for (uint8_t i = 0; i < DXL_COUNT; i++) {
        char key[8];
        snprintf(key, sizeof(key), "zero%u", DXL_IDS[i]);
        // 未較正のキーを getFloat すると Preferences がエラーを吐くので、先に有無を見る
        legZero[i] = prefs.isKey(key) ? prefs.getFloat(key) : 0.0f;
    }
    prefs.end();
    Serial.printf("crank zero: id%u=%.1f id%u=%.1f deg\n",
                  DXL_IDS[0], legZero[0], DXL_IDS[1], legZero[1]);
}

static void saveZeros() {
    prefs.begin(NVS_NAMESPACE, false);
    for (uint8_t i = 0; i < legCount; i++) {
        char key[8];
        snprintf(key, sizeof(key), "zero%u", legIds[i]);
        prefs.putFloat(key, legZero[i]);
    }
    prefs.end();
}

// 動作モードを揃える。切り替えにはトルクを落とす必要があるので、変わるときだけ触る
static void setMode(uint8_t mode) {
    if (legMode == mode) return;
    for (uint8_t i = 0; i < legCount; i++) {
        dxl.torqueOff(legIds[i]);
        dxl.setOperatingMode(legIds[i], mode);
        dxl.writeControlTableItem(PROFILE_VELOCITY, legIds[i], DXL_PROFILE_VEL);
        dxl.writeControlTableItem(PROFILE_ACCELERATION, legIds[i], DXL_PROFILE_ACC);
        dxl.torqueOn(legIds[i]);
    }
    legMode = mode;
}

// 起動時は通信の確認と速度モードの設定だけ。回すのは A ボタンから
static void setupLegs() {
    loadZeros();
    Serial2.begin(DXL_BAUD, SERIAL_8N1, DXL_RX_PIN, DXL_TX_PIN);
    dxl = Dynamixel2Arduino(Serial2);
    dxl.begin(DXL_BAUD);
    dxl.setPortProtocolVersion(DXL_PROTOCOL);
    delay(DXL_SETTLE_MS);  // 開いた直後は最初のパケットが落ちることがある

    for (uint8_t i = 0; i < DXL_COUNT; i++) {
        // 取りこぼしでモータを見失わないよう数回試す
        for (uint8_t retry = 0; retry < DXL_PING_RETRY; retry++) {
            if (dxl.ping(DXL_IDS[i])) { legIds[legCount++] = DXL_IDS[i]; break; }
        }
    }
    if (legCount == 0) {
        Serial.println("No DYNAMIXEL found");
        M5.Display.println("No servos");
        return;
    }
    for (uint8_t i = 0; i < legCount; i++) {
        Serial.printf("DYNAMIXEL id=%u model=%u\n", legIds[i], dxl.getModelNumber(legIds[i]));
    }
    M5.Display.printf("Servos: %u\n", legCount);
    M5.Display.printf("A:run B:%d C:zero\n", (int)HOME_DEG);

    setMode(OP_VELOCITY);
    for (uint8_t i = 0; i < legCount; i++) dxl.setGoalVelocity(legIds[i], 0, UNIT_RPM);
}

// A ボタンで回転／停止をトグルする。両脚とも同じ向き・同じ速度で回す
static void toggleSpin() {
    if (legCount == 0) return;
    spinning = !spinning;
    setMode(OP_VELOCITY);  // 連続回転させるだけなので位置ではなく速度で回す
    for (uint8_t i = 0; i < legCount; i++) {
        dxl.setGoalVelocity(legIds[i], spinning ? SPIN_RPM : 0, UNIT_RPM);
    }
    Serial.println(spinning ? "spin: on" : "spin: off");
}

// 両脚を指定したクランク角へ動かして保持する。回転は止める
static void goCrank(float deg) {
    if (legCount == 0) return;
    spinning = false;
    for (uint8_t i = 0; i < legCount; i++) dxl.setGoalVelocity(legIds[i], 0, UNIT_RPM);

    // 現在角を読むのはモードを移した後。速度モードでは多回転ぶんの値が当てにならない
    setMode(OP_EXTENDED_POSITION);
    for (uint8_t i = 0; i < legCount; i++) {
        float present = dxl.getPresentPosition(legIds[i], UNIT_DEGREE);
        // 較正した原点から見て deg になる位置へ。近い側に回るので最大でも半回転
        float goal = present + wrapDeg(deg - crankDeg(i, present));
        dxl.setGoalPosition(legIds[i], goal, UNIT_DEGREE);
        Serial.printf("id=%u goto: crank %.1f -> %.1f deg\n",
                      legIds[i], crankDeg(i, present), deg);
    }
}

// B ボタンはホーム角へ
static void goHome() { goCrank(HOME_DEG); }

// C ボタン: クランク原点の較正。
// 1回目でトルクを抜いて手で回せるようにし、CAD組立姿勢に合わせて2回目を押すと
// その角度を原点として NVS に保存する。方策はこの原点を基準に学習されている
static void toggleCalibration() {
    if (legCount == 0) return;
    calibrating = !calibrating;

    if (calibrating) {
        spinning = false;
        // setMode は最後にトルクを入れる。その瞬間に古い目標へ飛ばないよう、
        // 先に目標を現在位置へ置いてから切り替える
        for (uint8_t i = 0; i < legCount; i++) {
            dxl.setGoalPosition(legIds[i],
                                dxl.getPresentPosition(legIds[i], UNIT_DEGREE),
                                UNIT_DEGREE);
        }
        setMode(OP_EXTENDED_POSITION);  // 位置を読む土俵に揃えてからトルクを抜く
        for (uint8_t i = 0; i < legCount; i++) dxl.torqueOff(legIds[i]);
        Serial.println("calibration: torque off, set the CAD assembly pose");
        return;
    }

    for (uint8_t i = 0; i < legCount; i++) {
        float present = dxl.getPresentPosition(legIds[i], UNIT_DEGREE);
        // いま作った姿勢が CALIB_POSE_DEG である、として原点を逆算する
        legZero[i] = present - CALIB_POSE_DEG;
        // トルクを入れた瞬間に飛ばないよう、目標は「いまの位置」。ここに legZero を
        // 渡すと CALIB_POSE_DEG ぶん離れた場所が目標になり、脚が 280 度回り出す。
        // 直前に方策を走らせているとプロファイルが切れていて、全速で回る
        dxl.setGoalPosition(legIds[i], present, UNIT_DEGREE);
        dxl.torqueOn(legIds[i]);
        Serial.printf("calibration: id=%u zero = %.1f deg (pose = %.1f)\n",
                      legIds[i], legZero[i], CALIB_POSE_DEG);
    }
    saveZeros();
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

    if (policyRunning) {
        canvas.setTextSize(2);
        canvas.setTextColor(COLOR_LABEL);
        canvas.setTextDatum(bottom_center);
        char line[48];
        snprintf(line, sizeof(line), "%s %+0.2f %+0.2f %luHz",
                 policyDryRun ? "DRY" : "RUN",
                 policyAction[0], policyAction[1], (unsigned long)policyHz);
        canvas.drawString(line, SCREEN_W / 2, SCREEN_H - STATUS_MARGIN);
        canvas.setTextDatum(top_left);
    }
    // サーボが空転していても回っているか分かるよう、状態を出しておく
    if (spinning) {
        canvas.setTextSize(2);
        canvas.setTextColor(COLOR_LABEL);
        canvas.setTextDatum(bottom_center);
        canvas.drawString("SPIN", SCREEN_W / 2, SCREEN_H - STATUS_MARGIN);
        canvas.setTextDatum(top_left);
    }

    canvas.pushSprite(0, 0);
}

// 較正中は顔の代わりにクランク角を出す。CAD組立姿勢に合わせる作業のための画面
static void drawCalibration() {
    canvas.fillSprite(COLOR_BG);
    canvas.setTextSize(3);
    canvas.setTextColor(COLOR_EYE);
    canvas.setCursor(8, 12);
    canvas.println("CALIBRATE");

    canvas.setTextSize(2);
    for (uint8_t i = 0; i < legCount; i++) {
        float present = dxl.getPresentPosition(legIds[i], UNIT_DEGREE);
        canvas.setCursor(8, 70 + i * 28);
        canvas.printf("id%u  crank %6.1f", legIds[i], crankDeg(i, present));
    }

    canvas.setTextColor(COLOR_LABEL);
    canvas.setCursor(8, 150);
    canvas.println("Both tips just");
    canvas.setCursor(8, 175);
    canvas.println("touching floor,");
    canvas.setCursor(8, 200);
    canvas.printf("then C (=%d deg)\n", (int)CALIB_POSE_DEG);
    canvas.pushSprite(0, 0);
}

// --- 方策の実行 (50Hz) ---

// --- 方策実行の記録（Sim2Real の食い違いを測るため）---
//
// 実機で観測と行動をそのまま残し、PC 側で同じ行動をシミュレータに流し込む。
// 同じ入力に対して応答が違えば、モデルのどこが実機と違うかが直接分かる。
// 推測ではなく測定で切り分けるための道具。
static const uint16_t LOG_SAMPLES = 500;        // 50Hz で 10 秒
static float    logObs[LOG_SAMPLES][GORON_OBS];
static float    logAct[LOG_SAMPLES][GORON_ACT];
static uint32_t logT[LOG_SAMPLES];
static volatile uint16_t logCount = 0;
static volatile bool     logging  = false;

static void dumpLog() {
    Serial.printf("L,begin,samples,%u\n", logCount);
    for (uint16_t i = 0; i < logCount; i++) {
        Serial.printf("L,%lu", (unsigned long)logT[i]);
        for (int j = 0; j < GORON_OBS; j++) Serial.printf(",%.4f", logObs[i][j]);
        for (int j = 0; j < GORON_ACT; j++) Serial.printf(",%.4f", logAct[i][j]);
        Serial.println();
    }
    Serial.println("L,end");
}

// 観測を組み立てる。並びはエクスポートしたヘッダの冒頭の表に従う
static void buildObs(float obs[GORON_OBS], float crank[DXL_COUNT]) {
    float a[3], g[3];
    M5.Imu.update();
    M5.Imu.getAccel(&a[0], &a[1], &a[2]);   // [g]、静止時は上方向を指す
    M5.Imu.getGyro(&g[0], &g[1], &g[2]);    // [deg/s]

    // IMU 軸 -> ボディ座標系。較正の残差 4.6 度ぶんもここで入る
    for (int i = 0; i < 3; i++) {
        float up = 0.0f, w = 0.0f;
        for (int j = 0; j < 3; j++) {
            up += GORON_IMU_TO_BODY[i][j] * a[j];
            w  += GORON_IMU_TO_BODY[i][j] * g[j];
        }
        obs[i]     = up;                       // [0:3] 重力方向（＝上方向）
        obs[3 + i] = w * (float)M_PI / 180.0f;  // [3:6] 角速度 [rad/s]
    }
    for (int i = 0; i < 3; i++) { lastImu[i] = a[i]; lastImu[3 + i] = g[i]; }

    for (uint8_t i = 0; i < legCount; i++) {
        float present = dxl.getPresentPosition(legIds[i], UNIT_DEGREE);
        crank[i] = present;
        // 方策が学習した原点へ合わせる。crawl/selfright は 0、forward は 240
        float q = (crankDeg(i, present) - GORON_CRANK_ZERO_DEG) * (float)M_PI / 180.0f;
        obs[6 + i] = sinf(q);
        obs[8 + i] = cosf(q);
        // 速度は 0.229 rpm 単位。無負荷速度で正規化する
        lastRaw[i] = present; lastCrank[i] = crankDeg(i, present);
        float vel = dxl.getPresentVelocity(legIds[i], UNIT_RPM) * 2.0f * (float)M_PI / 60.0f;
        obs[10 + i] = vel / GORON_NO_LOAD_SPEED;
        // トルクは電流から。全開PWMの実測電流を失速トルクに対応づけている
        float cur = (float)(int16_t)dxl.readControlTableItem(PRESENT_CURRENT, legIds[i]);
        obs[12 + i] = (cur / DXL_SAT_CURRENT_MA);
    }
}

static void stopPolicy(const char* why) {
    policyRunning = false;
    logging = false;
    for (uint8_t i = 0; i < legCount; i++) {
        dxl.setGoalPosition(legIds[i], dxl.getPresentPosition(legIds[i], UNIT_DEGREE),
                            UNIT_DEGREE);
    }
    Serial.printf("policy: stopped (%s)\n", why);
}

static void startPolicy(bool dry = false) {
    policyDryRun = dry;
    if (legCount < DXL_COUNT) {
        Serial.println("policy: needs both servos");
        return;
    }
    // 較正中はトルクが抜けている。そのまま走らせると、方策は計算も指令もするのに
    // 脚は一切動かず、記録には「速度も電流もゼロ」だけが残る。いちばん気づきにくい
    // 失敗の仕方なので、拒否する
    if (calibrating) {
        Serial.println("policy: still calibrating -- press C to finish first");
        return;
    }
    setMode(OP_EXTENDED_POSITION);
    // setMode はモードが変わらなければ何もしない。トルクだけは毎回入れ直す
    for (uint8_t i = 0; i < legCount; i++) {
        dxl.setGoalPosition(legIds[i], dxl.getPresentPosition(legIds[i], UNIT_DEGREE),
                            UNIT_DEGREE);
        dxl.torqueOn(legIds[i]);
    }
    for (uint8_t i = 0; i < legCount; i++) {
        // プロファイルを切る。入れるとサーボ同定と違う条件になる
        dxl.writeControlTableItem(PROFILE_VELOCITY, legIds[i], 0);
        dxl.writeControlTableItem(PROFILE_ACCELERATION, legIds[i], 0);
        policyTarget[i] = dxl.getPresentPosition(legIds[i], UNIT_DEGREE);
        dxl.setGoalPosition(legIds[i], policyTarget[i], UNIT_DEGREE);
    }
    legMode = 0xFF;          // プロファイルを触ったので次の setMode で入れ直させる
    policyLateUs = 0;
    policyRunning = true;
    // トルクを入れた直後に読み返すと反映前の値を拾うことがある。表示が当てに
    // ならないと確認の意味がなくなるので、少し置いてから読む
    delay(20);
    int torque = 0;
    for (uint8_t i = 0; i < legCount; i++) {
        torque += (int)dxl.readControlTableItem(TORQUE_ENABLE, legIds[i]);
    }
    Serial.printf("policy: started (%s, %d Hz, torque %d/%u)%s\n",
                  GORON_POLICY, GORON_CONTROL_HZ, torque, legCount,
                  dry ? " DRY RUN - servos will not move" : "");
}

static void policyStepOnce();


// Core1 で 50Hz を刻む。vTaskDelayUntil は周期の起点を保つので、
// 1回遅れても次で取り返し、平均周期がずれない
static void policyTask(void*) {
    const TickType_t period = pdMS_TO_TICKS(1000 / GORON_CONTROL_HZ);
    TickType_t last = xTaskGetTickCount();
    uint32_t count = 0, mark = millis();
    for (;;) {
        if (!policyRunning) {
            vTaskDelay(pdMS_TO_TICKS(20));
            last = xTaskGetTickCount();
            policyHz = 0;
            continue;
        }
        uint32_t t0 = micros();
        policyStepOnce();
        uint32_t took = micros() - t0;
        if (took > policyLateUs) policyLateUs = took;   // 1周期ぶんの処理時間
        if (++count >= GORON_CONTROL_HZ) {
            uint32_t now = millis();
            policyHz = count * 1000 / (now - mark);
            count = 0; mark = now;
        }
        vTaskDelayUntil(&last, period);
    }
}

// 1制御周期。観測 -> 方策 -> 目標角の更新
static void policyStepOnce() {
    float obs[GORON_OBS], crank[DXL_COUNT];
    buildObs(obs, crank);
    goron_policy(obs, policyAction);

    if (logging && logCount < LOG_SAMPLES) {
        logT[logCount] = micros();
        memcpy(logObs[logCount], obs, sizeof(obs));
        memcpy(logAct[logCount], policyAction, sizeof(policyAction));
        logCount++;
    }

    const float maxDeltaDeg = GORON_MAX_DELTA * 180.0f / (float)M_PI;
    const float maxLeadDeg  = GORON_MAX_LEAD  * 180.0f / (float)M_PI;
    for (uint8_t i = 0; i < legCount; i++) {
        policyTarget[i] += policyAction[i] * maxDeltaDeg;
        // 実測角から離れすぎないように締める。これが無いと引っかかったときに
        // 目標角が無限に先走り、復帰に数秒かかる
        float lo = crank[i] - maxLeadDeg, hi = crank[i] + maxLeadDeg;
        policyTarget[i] = policyTarget[i] < lo ? lo
                        : (policyTarget[i] > hi ? hi : policyTarget[i]);
        if (!policyDryRun) dxl.setGoalPosition(legIds[i], policyTarget[i], UNIT_DEGREE);
    }

    if (policyDryRun) {
        static uint32_t nextLog = 0;
        if (millis() >= nextLog) {
            nextLog = millis() + 500;
            Serial.printf("O,up %+.2f %+.2f %+.2f, w %+.2f %+.2f %+.2f, "
                          "sc %+.2f %+.2f %+.2f %+.2f, v %+.3f %+.3f, "
                          "t %+.3f %+.3f, act %+.3f %+.3f, %luHz step %luus\n",
                          obs[0], obs[1], obs[2], obs[3], obs[4], obs[5],
                          obs[6], obs[7], obs[8], obs[9], obs[10], obs[11],
                          obs[12], obs[13], policyAction[0], policyAction[1],
                          (unsigned long)policyHz, (unsigned long)policyLateUs);
        }
    }
}

// --- 滑走中の加速度を高速記録する（床摩擦の実測用）---
//
// 平らな床を滑っている間、水平方向に働くのは摩擦だけなので、加速度計が示す
// 「見かけの上方向」は鉛直から atan(mu) 傾く。傾斜台を使うのと同じ原理だが、
// 板を傾ける代わりに実際の床の上で滑らせられる。
// テレメトリ(20Hz)では滑走中に数点しか取れないため、専用の高速記録を用意する。
// 115200 baud では毎サンプル送ると帯域が足りないので、貯めてから吐く。
static const uint16_t ACC_SAMPLES = 2000;
static uint32_t accT[ACC_SAMPLES];
static int16_t  accXYZ[ACC_SAMPLES][3];

static void recordAccel(uint32_t ms) {
    // 要求された時間ぶんをバッファに収める。全速で回すとバッファが 0.45 秒で
    // 埋まりきり、「合図 -> 押す -> 滑る」が記録に入らない
    const uint32_t period = (ms * 1000UL) / ACC_SAMPLES;
    uint32_t t0 = micros();
    uint16_t n = 0;
    while (n < ACC_SAMPLES && (micros() - t0) < ms * 1000UL) {
        while ((micros() - t0) < (uint32_t)n * period) { /* 次の刻みまで待つ */ }
        float ax, ay, az;
        M5.Imu.update();
        M5.Imu.getAccel(&ax, &ay, &az);
        accT[n] = micros() - t0;
        accXYZ[n][0] = (int16_t)(ax * 10000.0f);   // 0.1 mg 単位に詰める
        accXYZ[n][1] = (int16_t)(ay * 10000.0f);
        accXYZ[n][2] = (int16_t)(az * 10000.0f);
        n++;
    }
    Serial.printf("A,begin,samples,%u,ms,%lu\n", n, (unsigned long)ms);
    for (uint16_t i = 0; i < n; i++) {
        Serial.printf("A,%lu,%d,%d,%d\n", (unsigned long)accT[i],
                      accXYZ[i][0], accXYZ[i][1], accXYZ[i][2]);
    }
    Serial.println("A,end");
}

// --- サーボ同定用のステップ応答計測 ---
//
// 方策は Extended Position の目標角を動かして脚を回す。その応答から kp・減衰・
// 慣性をまとめて同定するため、目標角を階段状に動かして高速に記録する。
// Profile Velocity/Acceleration を 0 にして台形加減速を切らないと、測れるのは
// サーボの動特性ではなくプロファイル生成器の設定値になってしまう。
static const uint16_t IDENT_SAMPLES = 1200;
// フェーズごとに枠を分ける。分けないと最初のフェーズがバッファを食い尽くす。
// 落ちるかどうかを見るだけのフェーズ0は短くてよく、応答を見るフェーズ1に厚く配る
static const uint16_t IDENT_N0 = 300;
static const uint32_t IDENT_MS0 = 600;
static const uint32_t IDENT_MS1 = 1200;

static uint32_t identT[IDENT_SAMPLES];
static float    identPos[IDENT_SAMPLES];
static int16_t  identCur[IDENT_SAMPLES];

// 1フェーズぶん、可能なかぎり速く記録する。戻り値は取れたサンプル数
static uint16_t identRecord(uint8_t id, uint16_t from, uint16_t limit, uint32_t ms) {
    uint32_t t0 = micros();
    uint16_t n = from;
    while (n < limit && (micros() - t0) < ms * 1000UL) {
        identT[n]   = micros() - t0;
        identPos[n] = dxl.getPresentPosition(id, UNIT_DEGREE);
        identCur[n] = (int16_t)dxl.readControlTableItem(PRESENT_CURRENT, id);
        n++;
    }
    return n;
}

// "E <id> <step_deg>": トルクオフで放置 -> トルクオンして目標角を階段状に動かす
static void identifyServo(uint8_t id, float stepDeg) {
    // 重力トルクは推定せず既知の項として扱いたいので、機体の向きも一緒に残す。
    // 実験中はテレメトリが止まるため、ここで1回読んでおく
    float ax = 0, ay = 0, az = 0;
    M5.Imu.getAccel(&ax, &ay, &az);
    Serial.printf("E,begin,id=%u,step=%.1f,crank=%.2f,accel=%.4f;%.4f;%.4f\n",
                  id, stepDeg, crankDeg(id == legIds[0] ? 0 : 1,
                                        dxl.getPresentPosition(id, UNIT_DEGREE)),
                  ax, ay, az);

    // フェーズ0: トルクを切って落ちるかを見る。ギヤの乾性摩擦の有無が分かる
    dxl.torqueOff(id);
    uint16_t n0 = identRecord(id, 0, IDENT_N0, IDENT_MS0);

    // フェーズ1: 駆動してステップ入力。プロファイルを切って純粋な階段にする
    float here = dxl.getPresentPosition(id, UNIT_DEGREE);
    dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);      // 0 = プロファイル無効
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.setGoalPosition(id, here, UNIT_DEGREE);
    dxl.torqueOn(id);
    delay(100);
    here = dxl.getPresentPosition(id, UNIT_DEGREE);
    dxl.setGoalPosition(id, here + stepDeg, UNIT_DEGREE);
    uint16_t n1 = identRecord(id, n0, IDENT_SAMPLES, IDENT_MS1);

    // 元の設定に戻す
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, DXL_PROFILE_VEL);
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, DXL_PROFILE_ACC);
    legMode = 0xFF;  // モードとプロファイルを触ったので、次の setMode で入れ直させる

    Serial.printf("E,phase0,%u,samples,%u\n", 0, n0);
    Serial.printf("E,phase1,%u,samples,%u,start,%.2f,goal,%.2f\n",
                  1, n1 - n0, here, here + stepDeg);
    for (uint16_t i = 0; i < n1; i++) {
        Serial.printf("E,%u,%lu,%.3f,%d\n", i < n0 ? 0 : 1,
                      (unsigned long)identT[i], identPos[i], identCur[i]);
    }
    Serial.println("E,end");
}

// PC からの設定ダンプコマンド: "D"
// RobotParams に推定値のまま残っている項目（供給電圧・内部ゲイン・各種上限）を
// 実機から読み出す。特に電圧は効く: XL330 の 0.52 N.m / 103 rpm は 5.0V での値
static void dumpControlTable() {
    for (uint8_t i = 0; i < legCount; i++) {
        uint8_t id = legIds[i];
        // 方策が動かないときの切り分け用。指令が届いているか、トルクが入っているか、
        // 目標と現在がどれだけ離れているかを、サーボ自身に聞く
        // 速度と電流は方策の観測 obs[10:14] の元になる量。既知の回転をさせながら
        // これを読めば、実機の観測がシミュレータと同じ意味かを確かめられる
        Serial.printf("D,id=%u,torque=%d,mode=%d,goal=%.1f,present=%.1f,"
                      "err=%.1f,vel_rpm=%.2f,cur=%d,moving=%d\n",
                      id,
                      (int)dxl.readControlTableItem(TORQUE_ENABLE, id),
                      (int)dxl.readControlTableItem(OPERATING_MODE, id),
                      dxl.readControlTableItem(GOAL_POSITION, id) * 0.087891f,
                      dxl.getPresentPosition(id, UNIT_DEGREE),
                      dxl.readControlTableItem(GOAL_POSITION, id) * 0.087891f
                        - dxl.getPresentPosition(id, UNIT_DEGREE),
                      dxl.getPresentVelocity(id, UNIT_RPM),
                      (int)(int16_t)dxl.readControlTableItem(PRESENT_CURRENT, id),
                      (int)dxl.readControlTableItem(MOVING, id));
        Serial.printf("D,id=%u,volt=%.1f,temp=%d,kp=%d,ki=%d,kd=%d,"
                      "cur_lim=%d,vel_lim=%lu,pwm_lim=%d,mode=%d,delay=%u\n",
                      id,
                      dxl.readControlTableItem(PRESENT_INPUT_VOLTAGE, id) * 0.1f,
                      (int)dxl.readControlTableItem(PRESENT_TEMPERATURE, id),
                      (int)dxl.readControlTableItem(POSITION_P_GAIN, id),
                      (int)dxl.readControlTableItem(POSITION_I_GAIN, id),
                      (int)dxl.readControlTableItem(POSITION_D_GAIN, id),
                      (int)dxl.readControlTableItem(CURRENT_LIMIT, id),
                      (unsigned long)dxl.readControlTableItem(VELOCITY_LIMIT, id),
                      (int)dxl.readControlTableItem(PWM_LIMIT, id),
                      (int)dxl.readControlTableItem(OPERATING_MODE, id),
                      (unsigned)(uint8_t)dxl.readControlTableItem(RETURN_DELAY_TIME, id));
    }
}

// PC からの原点補正コマンド: "Z <deg_id1> <deg_id2>"
// mirror.py の --crank-offset と同じ符号・同じ値を渡すと、その補正が焼き込まれる。
// crank = raw - zero なので、クランク角を d 動かすには zero を -d する
static void handleSerialCommand() {
    if (!Serial.available()) return;
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (legCount == 0) { Serial.println("no servos"); return; }
    if (line.startsWith("D")) { dumpControlTable(); return; }

    // "R": Return Delay Time を 0 にする。出荷値 250 (=500us) のままだと
    // 1トランザクションごとに 0.5ms 捨てることになり、高速記録も 50Hz 制御も苦しい
    if (line.startsWith("R")) {
        // EEPROM 項目なのでトルクを切らないと書き込みが拒否される
        for (uint8_t i = 0; i < legCount; i++) {
            dxl.torqueOff(legIds[i]);
            dxl.writeControlTableItem(RETURN_DELAY_TIME, legIds[i], 0);
            Serial.printf("id=%u return delay -> %u\n", legIds[i],
                          (unsigned)(uint8_t)dxl.readControlTableItem(RETURN_DELAY_TIME,
                                                                      legIds[i]));
        }
        legMode = 0xFF;  // トルクを落としたので、次の setMode で入れ直させる
        return;
    }

    // "P": 方策の起動/停止。"W": 試運転の連続回転（A ボタンから移設）
    if (line.startsWith("P")) {
        int dry = 0;
        bool hasArg = sscanf(line.c_str() + 1, "%d", &dry) == 1;
        if (policyRunning) stopPolicy("serial");
        else               startPolicy(hasArg && dry == 0);   // "P 0" = 空回し
        return;
    }

    // "L": 記録を開始（方策も一緒に走らせる）。"L?" で吐き出す
    if (line.startsWith("L")) {
        if (line.indexOf('?') >= 0) { dumpLog(); return; }
        logCount = 0;
        logging = true;
        if (!policyRunning) startPolicy(false);
        Serial.printf("L,recording %u samples (%.1f s)\n",
                      LOG_SAMPLES, (float)LOG_SAMPLES / GORON_CONTROL_HZ);
        return;
    }
    if (line.startsWith("W")) {
        if (policyRunning) stopPolicy("spin");
        toggleSpin();
        return;
    }

    // "A <ms>": 加速度を高速記録して吐く（床摩擦の滑走試験用）
    if (line.startsWith("A")) {
        int ms = 0;
        if (sscanf(line.c_str() + 1, "%d", &ms) != 1 || ms <= 0 || ms > 10000) {
            Serial.println("usage: A <ms>  (1..10000)");
            return;
        }
        recordAccel((uint32_t)ms);
        return;
    }

    // "G <deg>": 両脚を指定クランク角へ。実験の開始条件を揃えるため
    if (line.startsWith("G")) {
        float deg = 0.0f;
        if (sscanf(line.c_str() + 1, "%f", &deg) != 1) {
            Serial.println("usage: G <crank_deg>");
            return;
        }
        goCrank(deg);
        return;
    }

    // "E <id> <step_deg>": ステップ応答を記録して吐く
    if (line.startsWith("E")) {
        int id = 0; float step = 0.0f;
        if (sscanf(line.c_str() + 1, "%d %f", &id, &step) != 2) {
            Serial.println("usage: E <id> <step_deg>");
            return;
        }
        identifyServo((uint8_t)id, step);
        return;
    }

    // "S <deg_left> <deg_right>": いまの姿勢がこのクランク角だ、と教えて原点を決める
    if (line.startsWith("S")) {
        float a[DXL_COUNT] = {0.0f, 0.0f};
        if (sscanf(line.c_str() + 1, "%f %f", &a[0], &a[1]) != 2) {
            Serial.println("usage: S <deg_left> <deg_right>");
            return;
        }
        for (uint8_t i = 0; i < legCount; i++) {
            legZero[i] = dxl.getPresentPosition(legIds[i], UNIT_DEGREE) - a[i];
        }
        saveZeros();
        for (uint8_t i = 0; i < legCount; i++) {
            Serial.printf("crank zero: id%u=%.1f deg (pose = %.1f, saved)\n",
                          legIds[i], legZero[i], a[i]);
        }
        return;
    }
    if (!line.startsWith("Z")) return;

    float d[DXL_COUNT] = {0.0f, 0.0f};
    if (sscanf(line.c_str() + 1, "%f %f", &d[0], &d[1]) != 2) {
        Serial.println("usage: Z <deg_id1> <deg_id2>");
        return;
    }
    for (uint8_t i = 0; i < legCount; i++) legZero[i] -= d[i];
    saveZeros();
    for (uint8_t i = 0; i < legCount; i++) {
        Serial.printf("crank zero: id%u=%.1f deg (saved)\n", legIds[i], legZero[i]);
    }
}

// 実機の姿勢を PC 側 (scripts/mirror.py) へ流す。1行1サンプルの CSV。
// crank は較正した原点からの角度なので、そのまま sim の関節角として使える
static void sendTelemetry() {
    if (legCount == 0) return;

    float ax, ay, az, gx, gy, gz;
    float raw[DXL_COUNT], crank[DXL_COUNT];
    if (policyRunning) {
        // 制御ループが読んだ写しを使う。バスにも IMU にも触らない
        ax = lastImu[0]; ay = lastImu[1]; az = lastImu[2];
        gx = lastImu[3]; gy = lastImu[4]; gz = lastImu[5];
        for (uint8_t i = 0; i < legCount; i++) {
            raw[i] = lastRaw[i]; crank[i] = lastCrank[i];
        }
    } else {
        M5.Imu.getAccel(&ax, &ay, &az);  // [g]。静止時は重力方向を指す
        M5.Imu.getGyro(&gx, &gy, &gz);   // [deg/s]
        for (uint8_t i = 0; i < legCount; i++) {
            raw[i]   = dxl.getPresentPosition(legIds[i], UNIT_DEGREE);
            crank[i] = crankDeg(i, raw[i]);
        }
    }
    Serial.printf("T,%lu,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n",
                  (unsigned long)millis(), raw[0], raw[1], crank[0], crank[1],
                  ax, ay, az, gx, gy, gz);
}

// --- setup / loop ---

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    M5.Display.fillScreen(COLOR_BG);
    M5.Display.setTextSize(2);

    Serial.begin(115200);
    Serial.setTimeout(20);  // コマンド読みでループを止めないため既定の1秒から短くする
    while (!Serial && millis() < 3000);  // 起動直後のログを取りこぼさない（最大3秒待つ）

    setupLegs();
    delay(BOOT_SHOW_MS);
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

    // 制御は Core1 へ。優先度は描画より高くする
    xTaskCreatePinnedToCore(policyTask, "policy", 8192, nullptr, 2, nullptr, 1);

    randomSeed(esp_random());
    uint32_t now = millis();
    blinkAt = now + random(BLINK_MIN_MS, BLINK_MAX_MS);
    gazeAt  = now + random(SACCADE_MIN_MS, SACCADE_MAX_MS);
}

void loop() {
    M5.update();

    // フレーム待ちより先に見る。押した瞬間を取りこぼさないため。
    // 較正中は手で脚を動かしている最中なので、動かす操作は受け付けない。
    // A は実機運用の主操作なので方策の起動/停止に割り当て、試運転用の連続回転は
    // シリアルの "W" に移した
    if (M5.BtnA.wasPressed() && !calibrating) {
        policyRunning ? stopPolicy("button") : startPolicy();
    }
    if (M5.BtnB.wasPressed() && !calibrating) {
        if (policyRunning) stopPolicy("home");
        goHome();
    }
    if (M5.BtnC.wasPressed()) {
        if (policyRunning) stopPolicy("calibration");
        toggleCalibration();
    }

    // 制御は Core1 の policyTask が回している。ここでは何もしない

    handleSerialCommand();

    uint32_t now = millis();
    static uint32_t nextTelem = 0;
    // 方策実行中も送る。値は制御ループの写しなので、バスにも制御周期にも
    // 影響しない（scripts/mirror で走行中の姿勢を見るため）
    if (now >= nextTelem) {
        nextTelem = now + TELEM_MS;
        sendTelemetry();
    }

    static uint32_t nextFrame = 0;
    if (now < nextFrame) return;
    nextFrame = now + FRAME_MS;

    if (calibrating) drawCalibration();
    else             drawFace(now);
}
