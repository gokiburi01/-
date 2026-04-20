import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js";

let squatCount = 0;
let jumpCount = 0;

let stageSquat = "up";
let stageJump = "ground";

let initialAnkleY = null;
const jumpThresholdRatio = 0.12;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const warning = document.getElementById("warning");

document.getElementById("resetBtn").onclick = () => {
    squatCount = 0;
    jumpCount = 0;
};

function angle(a, b, c) {
    const ab = { x: a.x - b.x, y: a.y - b.y };
    const cb = { x: c.x - b.x, y: c.y - b.y };
    const dot = ab.x * cb.x + ab.y * cb.y;
    const mag1 = Math.hypot(ab.x, ab.y);
    const mag2 = Math.hypot(cb.x, cb.y);
    return Math.acos(dot / (mag1 * mag2 + 1e-6)) * (180 / Math.PI);
}

async function main() {
    // Pose 初期化
    const visionObj = await vision;
    const pose = await visionObj.PoseLandmarker.createFromOptions(visionObj, {
        baseOptions: {
            modelAssetPath:
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/pose_landmarker_heavy.task"
        },
        runningMode: "video",
        outputSegmentationMasks: false
    });

    // カメラ
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
    });
    video.srcObject = stream;

    video.onloadeddata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        loop();
    };

    function isFullyVisible(landmarks) {
        const needed = [0, 23, 24, 25, 26, 27, 28]; // 全身チェック用
        return needed.every(i => landmarks[i] && landmarks[i].visibility > 0.6);
    }

    function loop() {
        const now = performance.now();

        pose.detectForVideo(video, now, (result) => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            if (!result.landmarks || result.landmarks.length === 0) {
                warning.style.display = "block";
                requestAnimationFrame(loop);
                return;
            }

            const lm = result.landmarks[0];

            // 全身が映っているか
            if (!isFullyVisible(lm)) {
                warning.style.display = "block";
                requestAnimationFrame(loop);
                return;
            }
            warning.style.display = "none";

            // 必要ポイント
            const L_hip = lm[23];
            const R_hip = lm[24];
            const L_knee = lm[25];
            const R_knee = lm[26];
            const L_ankle = lm[27];
            const R_ankle = lm[28];

            // スクワット角度
            const L_angle = angle(L_hip, L_knee, L_ankle);
            const R_angle = angle(R_hip, R_knee, R_ankle);

            if (L_angle < 120 && R_angle < 120) {
                stageSquat = "down";
            }
            if (stageSquat === "down" && L_angle > 160 && R_angle > 160) {
                squatCount++;
                stageSquat = "up";
            }

            // ジャンプ
            const avgAnkleY = (L_ankle.y + R_ankle.y) / 2;

            if (initialAnkleY === null) initialAnkleY = avgAnkleY;

            const jumpUpBorder = initialAnkleY * (1 - jumpThresholdRatio);

            if (avgAnkleY < jumpUpBorder && stageJump === "ground") {
                stageJump = "air";
            }
            if (avgAnkleY >= initialAnkleY * 0.98 && stageJump === "air") {
                jumpCount++;
                stageJump = "ground";
            }

            // 骨格を描画（軽量版）
            ctx.fillStyle = "lime";
            lm.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, Math.PI * 2);
                ctx.fill();
            });

            // テキスト表示
            ctx.fillStyle = "yellow";
            ctx.font = "30px sans-serif";
            ctx.fillText("Squat: " + squatCount, 20, 40);
            ctx.fillText("Jump: " + jumpCount, 20, 80);

        });

        requestAnimationFrame(loop);
    }
}

main();
