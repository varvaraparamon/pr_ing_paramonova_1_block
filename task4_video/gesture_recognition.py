import argparse

import cv2
import mediapipe as mp

FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}


def count_extended_fingers(landmarks) -> int:
    """палец выпрямлен если его кончик выше среднего сустава"""
    return sum(
        1 for tip, pip in FINGERS.values() if landmarks[tip].y < landmarks[pip].y
    )


def thumb_extended(landmarks, handedness: str) -> bool:
    """большой палец отведён в сторону"""
    tip, ip = landmarks[4], landmarks[3]
    return tip.x > ip.x if handedness == "Left" else tip.x < ip.x


def classify_gesture(landmarks, handedness: str) -> str:
    extended = count_extended_fingers(landmarks)
    thumb = thumb_extended(landmarks, handedness)
    if extended == 0 and not thumb:
        return "fist"
    if extended >= 4:
        return "open palm"
    if extended == 0 and thumb:
        return "thumbs up"
    if (
        extended == 2
        and landmarks[8].y < landmarks[6].y
        and landmarks[12].y < landmarks[10].y
    ):
        return "peace"
    if extended == 1 and landmarks[8].y < landmarks[6].y:
        return "point"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="распознавание жестов руки на видео")
    parser.add_argument("video", nargs="?", help="путь к видеофайлу")
    parser.add_argument("--camera", type=int, default=None, help="индекс веб-камеры")
    parser.add_argument("-o", "--output", help="путь для сохранения видео с разметкой")
    args = parser.parse_args()

    source = args.camera if args.camera is not None else args.video
    if source is None:
        parser.error("укажите видеофайл или --camera")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"не удалось открыть источник видео: {source}")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    draw = mp.solutions.drawing_utils

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for lm, handed in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handed.classification[0].label
                gesture = classify_gesture(lm.landmark, label)
                draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.putText(
                    frame,
                    gesture,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                print(f"gesture: {gesture}")
        if writer:
            writer.write(frame)
        if args.camera is not None:
            cv2.imshow("gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
