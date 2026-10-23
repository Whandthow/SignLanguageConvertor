import collections, datetime
import cv2, joblib
import mediapipe as mp
from features import normalize

model = joblib.load("models/best.joblib")
hands = mp.solutions.hands.Hands(max_num_hands=1,
                                 min_detection_confidence=0.6,
                                 min_tracking_confidence=0.5)
drawer = mp.solutions.drawing_utils

WINDOW, NEED = 10, 7                 # літера підтверджується, якщо ≥7 з 10 кадрів
votes = collections.deque(maxlen=WINDOW)
last_label = None
text = ""

log = open("output.txt", "a", encoding="utf-8")
log.write(f"\n--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ---\n")
log.flush()
print("Розпізнавання жестів. Вихід: Esc у вікні камери.\n")

def emit(label):
    global text
    ch = " " if label == "_" else label
    text += ch
    print(ch, end="", flush=True)    # консоль
    log.write(ch); log.flush()       # паралельно в файл

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        flat = [c for p in hand.landmark for c in (p.x, p.y, p.z)]
        votes.append(model.predict([normalize(flat)])[0])

        if len(votes) == WINDOW:
            label, cnt = collections.Counter(votes).most_common(1)[0]
            if cnt >= NEED and label != last_label:
                emit(label)
                last_label = label
    else:
        votes.clear()
        last_label = None            # рука зникла → можна повторити ту саму літеру

    cv2.putText(frame, text[-30:], (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Sign2Text", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

print()
log.close(); cap.release(); cv2.destroyAllWindows()