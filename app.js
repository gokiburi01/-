const INTRO_COUNTDOWN = 10, MEMORY_TIME = 15, MEMORY_LENGTH = 20;
// 各運動は40秒のままにし、合計トレーニング時間を5分(300秒)に調整する
const EXERCISE_TIME = 40, TOTAL_TRAINING_TIME = 300, GOAL_REPS = 15, SCORE_THRESHOLD = 0.3;
// 描画・姿勢推定は端末性能によらず最大30fpsにそろえる。手の推定は十分な判定精度を
// 保てる12.5fpsに限定し、グーパー運動だけが重くなるのを防ぐ。
const TARGET_FPS = 30, FRAME_INTERVAL = 1000 / TARGET_FPS, HAND_INTERVAL = 80;
const $ = (id) => document.getElementById(id);
const setupScreen = $("setupScreen"), loadingScreen = $("loadingScreen"), countdownScreen = $("countdownScreen");
const memoryScreen = $("memoryScreen"), answerScreen = $("answerScreen"), beforeMemoryResultScreen = $("beforeMemoryResultScreen");
const trainingScreen = $("trainingScreen"), resultScreen = $("resultScreen");
const screens = [setupScreen, loadingScreen, countdownScreen, memoryScreen, answerScreen, beforeMemoryResultScreen, trainingScreen, resultScreen];
const heightInput = $("heightInput"), weightInput = $("weightInput"), genderInput = $("genderInput"), ageGroupInput = $("ageGroup");
const startBtn = $("startBtn"), loadingText = $("loadingText"), countdownNumber = $("countdownNumber");
const memoryDigits = $("memoryDigits"), memoryTimer = $("memoryTimer"), memoryAnswerInput = $("memoryAnswerInput");
const submitAnswerBtn = $("submitAnswer"), giveUpBtn = $("giveUpBtn"), beforeRate = $("beforeRate"), beforeCorrect = $("beforeCorrect");
const startTrainingBtn = $("startTrainingBtn"), video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
const warning = $("warning"), exerciseName = $("exerciseName"), exerciseTarget = $("exerciseTarget"), progressText = $("progressText");
const exerciseGoal = $("exerciseGoal");
const sq = $("sq"), jp = $("jp"), kcal = $("kcal"), fpsValue = $("fpsValue"), resetBtn = $("resetBtn"), endTrainingBtn = $("endTrainingBtn");
const excellentBadge = $("excellentBadge");
const beforeCorrectResult = $("beforeCorrectResult"), beforeRateResult = $("beforeRateResult");
const afterCorrectResult = $("afterCorrectResult"), afterRateResult = $("afterRateResult"), improveRate = $("improveRate");
const resultSquat = $("resultSquat"), resultJump = $("resultJump"), resultKcal = $("resultKcal"), restartBtn = $("restartBtn");

const exercises = [
    { name: "グーパー運動", type: "grip" }, { name: "もも上げ", type: "highKnee" },
    { name: "スクワット", type: "squat" }, { name: "ジャンプ", type: "jump" }
];
const skeleton = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
let phase = "before", randomDigits = "";
let beforeCorrectCount = 0, afterCorrectCount = 0, beforeScore = 0, afterScore = 0, improveScore = 0;
let detector = null, handDetector = null, cameraStream = null, running = false;
let gripCount = 0, highKneeCount = 0, squatCount = 0, jumpCount = 0, calorie = 0, currentExercise = 0, completedExercises = 0, remainExerciseTime = EXERCISE_TIME;
let squatState = "UP", kneeState = false, jumpCooldown = 0, prevHipY = null;
let handLandmarks = [], handDetectionPending = false, lastHandDetectionAt = 0, handResultVersion = 0, processedHandResultVersion = 0, gripPhase = "closed", gripPhaseEndsAt = 0, gripPhaseValidated = false, gripStableFrames = 0;
let countdownTimer = null, memoryTimerId = null, trainingTimer = null, animationId = null, fpsFrame = 0, lastFpsTime = performance.now(), lastInferenceAt = 0;
// トレーニング時間管理
const FULL_EXERCISES = Math.floor(TOTAL_TRAINING_TIME / EXERCISE_TIME);
const LAST_EXERCISE_DURATION = TOTAL_TRAINING_TIME - FULL_EXERCISES * EXERCISE_TIME; // 余り秒数（0なら無し）
// 上限を7セットに制限（ユーザーの指定）
let TOTAL_EXERCISES = Math.min(7, FULL_EXERCISES + (LAST_EXERCISE_DURATION > 0 ? 1 : 0));
let elapsedTraining = 0; // 秒単位で経過時間を管理
let excellentTimer = null;
let trainingSessionId = 0, trainingPreparationId = 0, exercisePreparing = false;

function showScreen(screen) {
    document.querySelectorAll(".screen").forEach((item) => { item.classList.add("hidden"); item.setAttribute("aria-hidden", "true"); });
    screen.classList.remove("hidden"); screen.removeAttribute("aria-hidden");
    window.MemoUI?.screenChanged(screen);
}
function clearTimers() {
    window.MemoUI?.cancelIntro();
    clearInterval(countdownTimer); clearInterval(memoryTimerId); clearInterval(trainingTimer);
    countdownTimer = memoryTimerId = trainingTimer = null;
}
function resetMemory() {
    phase = "before"; randomDigits = "";
    beforeCorrectCount = afterCorrectCount = beforeScore = afterScore = improveScore = 0;
    memoryAnswerInput.value = "";
}
function resetTraining() {
    trainingSessionId += 1; exercisePreparing = false;
    running = false; gripCount = highKneeCount = squatCount = jumpCount = calorie = currentExercise = completedExercises = 0;
    remainExerciseTime = EXERCISE_TIME; squatState = "UP"; kneeState = false; jumpCooldown = 0; prevHipY = null;
    handLandmarks = []; handDetectionPending = false; lastHandDetectionAt = 0; handResultVersion = processedHandResultVersion = 0; gripStableFrames = 0;
    elapsedTraining = 0;
    // 終了ボタンを再有効化（前回の無効化が残らないように）
    try { if (endTrainingBtn) { endTrainingBtn.disabled = false; endTrainingBtn.style.opacity = "1"; } } catch (e) {}
    updateTrainingUI();
}
function updateTrainingUI() {
    sq.textContent = squatCount; jp.textContent = jumpCount; kcal.textContent = calorie.toFixed(1);
    window.MemoUI?.updateLive();
}
function updateResultUI() {
    beforeCorrectResult.textContent = beforeCorrectCount + " / " + MEMORY_LENGTH;
    beforeRateResult.textContent = beforeScore + "%"; afterCorrectResult.textContent = afterCorrectCount + " / " + MEMORY_LENGTH;
    afterRateResult.textContent = afterScore + "%"; improveRate.textContent = (improveScore >= 0 ? "+" : "") + improveScore;
    resultSquat.textContent = squatCount + "回"; resultJump.textContent = jumpCount + "回"; resultKcal.textContent = calorie.toFixed(1) + " kcal";
}
function generateDigits() { return Array.from({ length: MEMORY_LENGTH }, () => Math.floor(Math.random() * 10)).join(""); }
function countCorrect(answer, correct) {
    return Array.from({ length: MEMORY_LENGTH }, (_, index) => answer[index] === correct[index]).filter(Boolean).length;
}

function startApp() { clearTimers(); resetMemory(); resetTraining(); randomDigits = generateDigits(); startCountdown(); }
function startAppWrapped() { try { console.log('startApp invoked'); startApp(); } catch (e) { console.error('startApp error', e); throw e; } }
function startCountdown() {
    showScreen(countdownScreen); let seconds = INTRO_COUNTDOWN; countdownNumber.textContent = seconds; window.MemoUI?.countdownTick(seconds);
    countdownTimer = setInterval(() => {
        countdownNumber.textContent = --seconds; window.MemoUI?.countdownTick(seconds);
        if (seconds <= 0) { clearInterval(countdownTimer); countdownTimer = null; startMemory(); }
    }, 1000);
}
function startMemory() {
    showScreen(memoryScreen);
    if (window.MemoUI) window.MemoUI.renderDigits(randomDigits);
    else memoryDigits.textContent = randomDigits.slice(0, 10) + "\n" + randomDigits.slice(10);
    let seconds = MEMORY_TIME; memoryTimer.textContent = seconds; window.MemoUI?.memoryTick(seconds);
    memoryTimerId = setInterval(() => {
        memoryTimer.textContent = --seconds; window.MemoUI?.memoryTick(seconds);
        if (seconds <= 0) { clearInterval(memoryTimerId); memoryTimerId = null; showAnswer(); }
    }, 1000);
}
function showAnswer() { memoryAnswerInput.value = ""; showScreen(answerScreen); memoryAnswerInput.focus({ preventScroll: true }); }
function submitMemory(giveUp = false) {
    if (answerScreen.classList.contains("hidden")) return;
    const answer = giveUp ? "" : String(memoryAnswerInput.value).replace(/\D/g, "");
    const correct = countCorrect(answer, randomDigits), score = Math.round(correct * 100 / MEMORY_LENGTH);
    if (phase === "before") {
        beforeCorrectCount = correct; beforeScore = score; beforeRate.textContent = score;
        beforeCorrect.textContent = correct + " / " + MEMORY_LENGTH; showScreen(beforeMemoryResultScreen); return;
    }
    afterCorrectCount = correct; afterScore = score; improveScore = afterScore - beforeScore; updateResultUI(); stopCamera(); showScreen(resultScreen); window.MemoUI?.saveResult();
}

async function prepareTraining() {
    const preparation = ++trainingPreparationId;
    showScreen(loadingScreen);
    try {
        loadingText.textContent = "カメラを起動しています…"; await setupCamera(preparation);
        if (preparation !== trainingPreparationId) return;
        loadingText.textContent = "運動を見守るAIを読み込んでいます…"; await ensureTrainingLibraries();
        if (preparation !== trainingPreparationId) return;
        await setupDetector();
        if (preparation !== trainingPreparationId) return;
        await setupHandDetector();
        if (preparation !== trainingPreparationId) return;
        startTraining();
    } catch (error) {
        if (preparation !== trainingPreparationId) return;
        stopCamera();
        console.error("トレーニングの準備に失敗しました", error);
        loadingText.textContent = "カメラまたはAIを開始できませんでした。カメラの許可と通信状態を確認して、もう一度お試しください。";
        window.MemoUI?.loadingFailed();
    }
}
function cancelPreparation() {
    trainingPreparationId += 1; stopCamera(); showScreen(beforeMemoryResultScreen);
}
async function setupCamera(preparation = trainingPreparationId) {
    if (cameraStream) return;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("このブラウザはカメラに対応していません");
    const acquiredStream = await navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: "user",
            // 高解像度の入力は推論量を大きく増やすため、運動判定に十分なVGA/30fpsを要求する。
            width: { ideal: 640, max: 640 }, height: { ideal: 480, max: 480 },
            frameRate: { ideal: TARGET_FPS, max: TARGET_FPS }
        },
        audio: false
    });
    if (preparation !== trainingPreparationId) {
        acquiredStream.getTracks().forEach((track) => track.stop());
        throw new DOMException("準備が中止されました", "AbortError");
    }
    cameraStream = acquiredStream;
    video.srcObject = cameraStream;
    await new Promise((resolve, reject) => {
        video.onloadedmetadata = async () => {
            try {
                await video.play(); canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480; resolve();
            } catch (error) { reject(error); }
        };
        video.onerror = reject;
    });
}
// カメラを使う段階でのみ読み込む。失敗したURLは再試行できます。
const libraryLoads = new Map();
function loadLibrary(url) {
    if (libraryLoads.has(url)) return libraryLoads.get(url);
    const promise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        let timer;
        const fail = () => { clearTimeout(timer); script.remove(); libraryLoads.delete(url); reject(new Error("AIライブラリを読み込めませんでした")); };
        script.src = url; script.async = true;
        script.onload = () => { clearTimeout(timer); resolve(); };
        script.onerror = fail;
        timer = setTimeout(fail, 30000);
        document.head.append(script);
    });
    libraryLoads.set(url, promise); return promise;
}
async function ensureTrainingLibraries() {
    if (!window.tf) await loadLibrary("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0");
    if (!window.poseDetection) await loadLibrary("https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection");
    if (!window.Hands) await loadLibrary("https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js");
}
async function setupDetector() {
    if (detector) return;
    if (!window.tf || !window.poseDetection) throw new Error("MoveNetの読み込みに失敗しました");
    await tf.setBackend("webgl"); await tf.ready();
    detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING });
}
async function setupHandDetector() {
    if (handDetector) return;
    if (!window.Hands) throw new Error("手の検出AIの読み込みに失敗しました");
    handDetector = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
    handDetector.setOptions({
        maxNumHands: 2,
        modelComplexity: 0,
        minDetectionConfidence: 0.75,
        minTrackingConfidence: 0.75,
        selfieMode: true
    });
    handDetector.onResults((results) => {
        handLandmarks = results.multiHandLandmarks || [];
        handResultVersion += 1;
        handDetectionPending = false;
    });
}
function startTraining() {
    trainingSessionId += 1; showScreen(trainingScreen); running = true; fpsFrame = 0; lastFpsTime = performance.now(); lastInferenceAt = 0; startExercise();
    animationId = requestAnimationFrame(poseLoop);
}
async function startExercise() {
    const session = trainingSessionId; exercisePreparing = true;
    const exercise = exercises[currentExercise]; remainExerciseTime = EXERCISE_TIME; exerciseName.textContent = exercise.name;
    // 各セットの秒数は基本40秒。ただし合計でちょうどTOTAL_TRAINING_TIMEになるよう
    // 最後のセットは余り秒数（LAST_EXERCISE_DURATION）があればそれを使う。
    const lastDur = LAST_EXERCISE_DURATION || EXERCISE_TIME;
    remainExerciseTime = (currentExercise < FULL_EXERCISES) ? EXERCISE_TIME : lastDur;
    window.MemoUI?.exerciseChanged(exercise);
    if (window.MemoUI) {
        const ready = await window.MemoUI.introduceExercise(exercise);
        if (!ready || !running || session !== trainingSessionId) return;
    }
    exercisePreparing = false;
    if (exercise.type === "grip") {
        gripPhase = "closed"; gripPhaseEndsAt = performance.now() + 5000;
        gripPhaseValidated = false; gripStableFrames = 0; processedHandResultVersion = handResultVersion; updateGripInstruction();
    }
    updateExerciseGoal();
    progressText.textContent = (completedExercises + 1) + " / " + TOTAL_EXERCISES + " セット";
    if (exercise.type !== "grip") exerciseTarget.textContent = "残り " + remainExerciseTime + " 秒";
    clearInterval(trainingTimer);
    trainingTimer = setInterval(() => {
        // 1秒経過
        remainExerciseTime -= 1;
        elapsedTraining += 1; window.MemoUI?.trainingTick();
        // グーパーは独自ロジックで更新
        if (exercise.type === "grip") updateGripPhase(performance.now());
        else exerciseTarget.textContent = "残り " + remainExerciseTime + " 秒";
        // 総合時間に到達したら終了（できるだけ正確に）
        if (elapsedTraining >= TOTAL_TRAINING_TIME) { finishTraining(); return; }
        if (remainExerciseTime <= 0) nextExercise();
    }, 1000);
}
function nextExercise() {
    clearInterval(trainingTimer); completedExercises += 1;
    if (completedExercises >= TOTAL_EXERCISES) { finishTraining(); return; }
    currentExercise = (currentExercise + 1) % exercises.length; startExercise();
}
function finishTraining() {
    trainingSessionId += 1; exercisePreparing = false;
    clearTimers(); running = false; cancelAnimationFrame(animationId); stopCamera(); phase = "after"; startCountdown();
}

function showWarning(message) {
    if (warning.textContent !== message) warning.textContent = message;
    warning.style.display = message ? "block" : "none";
}
function isFullBodyVisible(points) {
    return [0,5,6,11,12,13,14,15,16].every((index) => points[index]?.score >= SCORE_THRESHOLD);
}
function isGripPoseVisible(points) {
    return [5,6,9,10].every((index) => points[index]?.score >= SCORE_THRESHOLD);
}
async function poseLoop(now = performance.now()) {
    if (!running) return;
    if (exercisePreparing) { animationId = requestAnimationFrame(poseLoop); return; }
    if (now - lastInferenceAt < FRAME_INTERVAL) {
        animationId = requestAnimationFrame(poseLoop);
        return;
    }
    lastInferenceAt = now;
    try {
        const poses = await detector.estimatePoses(video); if (!running) return; drawCamera();
        if (exercises[currentExercise]?.type === "grip") requestHandDetection();
        if (!poses.length) showWarning("人物を検出できません");
        else if (exercises[currentExercise]?.type === "grip" && !isGripPoseVisible(poses[0].keypoints)) showWarning("両手を肩の前でカメラに向けてください");
        else if (!isFullBodyVisible(poses[0].keypoints)) showWarning("全身が画面に入る位置へ移動してください");
        else { showWarning(""); drawSkeleton(poses[0].keypoints); executeExercise(poses[0].keypoints); }
        updateTrainingUI(); updateFPS();
    } catch (error) { console.error("姿勢推定に失敗しました", error); }
    if (running) animationId = requestAnimationFrame(poseLoop);
}
function drawCamera() { ctx.save(); ctx.translate(canvas.width, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, canvas.width, canvas.height); ctx.restore(); }
function drawSkeleton(points) {
    ctx.lineWidth = 4; ctx.strokeStyle = "#00e5ff";
    skeleton.forEach(([a,b]) => {
        const first = points[a], second = points[b];
        if (first.score < SCORE_THRESHOLD || second.score < SCORE_THRESHOLD) return;
        ctx.beginPath(); ctx.moveTo(canvas.width - first.x, first.y); ctx.lineTo(canvas.width - second.x, second.y); ctx.stroke();
    });
}
function executeExercise(points) {
    switch (exercises[currentExercise].type) {
        case "grip": {
            // 同じHands結果を30fpsの姿勢ループで繰り返し数えない。
            if (handResultVersion !== processedHandResultVersion) {
                processedHandResultVersion = handResultVersion;
                detectGrip();
            }
            break;
        }
        case "highKnee": detectHighKnee(points); break;
        case "squat": detectSquat(points); break; case "jump": detectJump(points); break;
    }
    updateExerciseGoal();
}
function currentExerciseCount() {
    switch (exercises[currentExercise].type) {
        case "grip":
            return gripCount;

        case "highKnee":
            return highKneeCount;

        case "squat":
            return squatCount;

        case "jump":
            return jumpCount;

        default:
            return 0;
    }
}

function updateExerciseGoal() {
    const count = currentExerciseCount();

    const goalText = count >= GOAL_REPS
            ? "クリア！ " + count + "回　そのまま続けてもよいです"
            : "目標 " + GOAL_REPS + "回　現在 " + count + "回";
    if (exerciseGoal.textContent !== goalText) exerciseGoal.textContent = goalText;

    exerciseGoal.classList.toggle(
        "isCleared",
        count >= GOAL_REPS
    );
}
    

function getAngle(a, b, c) {
    const ab = { x: a.x - b.x, y: a.y - b.y }, cb = { x: c.x - b.x, y: c.y - b.y };
    const size = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
    return size ? Math.acos(Math.max(-1, Math.min(1, (ab.x * cb.x + ab.y * cb.y) / size))) * 180 / Math.PI : 180;
}
function detectSquat(points) {
    const [hip,knee,ankle] = [points[11],points[13],points[15]];
    if (![hip,knee,ankle].every((point) => point.score >= SCORE_THRESHOLD)) return;
    const angle = getAngle(hip,knee,ankle);
    if (squatState === "UP" && angle < 100) squatState = "DOWN";
    if (squatState === "DOWN" && angle > 160) { squatState = "UP"; squatCount += 1; calorie += 0.32; }
}
function detectJump(points) {
    const hip = points[11]; if (hip.score < SCORE_THRESHOLD) return;
    // 肩と腰の差から胴体高さを推定し、閾値を体格に応じて決定する
    const sL = points[5], sR = points[6];
    let torso = 100;
    if ((sL?.score ?? 0) > 0 || (sR?.score ?? 0) > 0) {
        const shoulderY = ((sL?.y || 0) + (sR?.y || 0)) / ((sL?.y ? 1 : 0) + (sR?.y ? 1 : 0) || 1);
        torso = Math.abs(shoulderY - hip.y) || torso;
    }
    const threshold = Math.max(20, torso * 0.12); // 胴体高さの12% または 20px の大きい方

    if (prevHipY === null) { prevHipY = hip.y; return; }
    if (jumpCooldown > 0) { jumpCooldown -= 1; prevHipY = hip.y; return; }
    // 上方向への急激な移動を検出（画面Yは上が小さい）
    if (prevHipY - hip.y > threshold) {
        jumpCount += 1; calorie += 0.45;
        jumpCooldown = Math.round(TARGET_FPS * 0.5); // 約0.5秒のクールダウン
        showExcellent();
    }
    prevHipY = hip.y;
}

function showExcellent() {
    if (!excellentBadge) return;
    try { clearTimeout(excellentTimer); } catch (e) {}
    window.MemoUI?.excellent();
    excellentBadge.style.display = "block";
    excellentTimer = setTimeout(() => { excellentBadge.style.display = "none"; excellentTimer = null; }, 1500);
}
function requestHandDetection() {
    const now = performance.now();
    if (!handDetector || handDetectionPending || now - lastHandDetectionAt < HAND_INTERVAL) return;
    lastHandDetectionAt = now;
    handDetectionPending = true;
    handDetector.send({ image: video }).catch((error) => {
        console.error("手の検出に失敗しました", error);
        handDetectionPending = false;
    });
}
function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function handGesture(landmarks) {
    // 手の大きさに対する指先距離と、4本の指の第2関節の角度を組み合わせる。
    // どちらか一方だけでは誤判定しやすいため、両方を満たす時だけ確定する。
    const palmWidth = Math.max(distance(landmarks[5], landmarks[17]), 0.001);
    const fingers = [[5,6,8], [9,10,12], [13,14,16], [17,18,20]];
    const tipDistance = fingers.reduce((sum, [, , tip]) => sum + distance(landmarks[tip], landmarks[0]), 0) / fingers.length / palmWidth;
    const angles = fingers.map(([base, joint, tip]) => getAngle(landmarks[base], landmarks[joint], landmarks[tip]));
    const bentFingers = angles.filter((angle) => angle < 125).length;
    const straightFingers = angles.filter((angle) => angle > 150).length;
    // 閾値をやや緩めて検出を改善（誤検知を防ぐため最低条件は維持）
    if (tipDistance < 1.6 && bentFingers >= 2) return "closed";
    if (tipDistance > 1.8 && straightFingers >= 2) return "open";
    return "unknown";
}
function detectGrip() {
    if (handLandmarks.length === 0) {
        gripStableFrames = 0;
        showWarning("手が映っていません。カメラの前で両手を前に出してください");
        return;
    }
    const states = handLandmarks.map(handGesture);
    const expected = gripPhase;
    // 2手検出時は両手が期待動作であることを要求。片手しか見えない場合は安定フレーム数を長めにして判定。
    if (handLandmarks.length >= 2) {
        if (states.every((state) => state === expected)) {
            gripStableFrames += 1;
            if (gripStableFrames >= 3) gripPhaseValidated = true;
        } else {
            gripStableFrames = 0;
            showWarning(gripPhase === "closed" ? "両手をしっかり握ってください" : "両手の指を大きく開いてください");
        }
    } else {
        // 片手しか映らない場合は安定判定をより厳密に（約0.4〜0.5秒）
        if (states[0] === expected) {
            gripStableFrames += 1;
            if (gripStableFrames >= 5) gripPhaseValidated = true;
        } else {
            gripStableFrames = 0;
            showWarning(gripPhase === "closed" ? "手をしっかり握ってください" : "指を大きく開いてください");
        }
    }
}
function updateGripInstruction() {
    const remaining = Math.max(0, Math.ceil((gripPhaseEndsAt - performance.now()) / 1000));
    const instruction = gripPhase === "closed" ? "握ってください" : "手を開いてください";
    exerciseTarget.textContent = `${instruction}（あと ${remaining} 秒）`;
    exerciseTarget.classList.toggle("openHand", gripPhase === "open");
}
function updateGripPhase(now) {
    if (exercises[currentExercise]?.type !== "grip") return;
    while (now >= gripPhaseEndsAt) {
        if (gripPhase === "open" && gripPhaseValidated) { gripCount += 1; calorie += 0.02; }
        gripPhase = gripPhase === "closed" ? "open" : "closed";
        gripPhaseEndsAt += 5000;
        gripPhaseValidated = false; gripStableFrames = 0;
        updateExerciseGoal();
    }
    updateGripInstruction();
}
function detectHighKnee(points) {
    const [hip,knee] = [points[11],points[13]];
    if (![hip,knee].every((point) => point.score >= SCORE_THRESHOLD)) return;
    if (knee.y < hip.y && !kneeState) { kneeState = true; highKneeCount += 1; calorie += 0.05; }
    if (knee.y > hip.y) kneeState = false;
}
function updateFPS() {
    fpsFrame += 1; const now = performance.now();
    if (now - lastFpsTime >= 1000) { fpsValue.textContent = Math.round(fpsFrame * 1000 / (now - lastFpsTime)); fpsFrame = 0; lastFpsTime = now; }
}
function stopCamera() {
    cameraStream?.getTracks().forEach((track) => track.stop()); cameraStream = null; video.srcObject = null;
}
function restartApp() {
    trainingPreparationId += 1;
    clearTimers(); running = false; cancelAnimationFrame(animationId); stopCamera(); resetMemory(); resetTraining();
    beforeRate.textContent = "0"; beforeCorrect.textContent = "0 / " + MEMORY_LENGTH; updateResultUI(); showScreen($("homeScreen"));
}

startBtn.addEventListener("click", startApp);
submitAnswerBtn.addEventListener("click", () => submitMemory(false));
giveUpBtn.addEventListener("click", () => submitMemory(true));
memoryAnswerInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.isComposing) submitMemory(false); });
startTrainingBtn.addEventListener("click", prepareTraining);
resetBtn.addEventListener("click", restartApp);
// トレーニング中に終了して最後の記憶テストへスキップする
if (endTrainingBtn) {
    const endHandler = (ev) => {
        ev.preventDefault();
        if (!running) return;
        // 二重押下を防ぐ
        endTrainingBtn.disabled = true;
        endTrainingBtn.style.opacity = "0.7";
        finishTraining();
    };
    endTrainingBtn.addEventListener("click", endHandler);

}
