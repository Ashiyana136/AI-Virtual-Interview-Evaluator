
import streamlit as st
import cv2
import mediapipe as mp
import math
import time

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="AI Virtual Interview Coach", layout="wide")
st.title("🧠 AI Virtual Interview Coach Dashboard")

# ==========================================================
# SESSION STATE
# ==========================================================
for key in ["run", "confidence_history", "emotion_history", "eye_history", "posture_history", "start_time", "interview_done"]:
    if key not in st.session_state:
        st.session_state[key] = False if key=="run" or key=="interview_done" else []

if "start_time" not in st.session_state:
    st.session_state.start_time = 0

# ==========================================================
# BUTTON FUNCTIONS
# ==========================================================
def start_interview():
    st.session_state.run = True
    st.session_state.interview_done = False
    st.session_state.confidence_history = []
    st.session_state.emotion_history = []
    st.session_state.eye_history = []
    st.session_state.posture_history = []
    st.session_state.start_time = time.time()

def stop_interview():
    st.session_state.run = False
    st.session_state.interview_done = True

col1, col2 = st.columns(2)
col1.button("🎥 Start Interview", on_click=start_interview)
col2.button("🛑 Stop Interview", on_click=stop_interview)

# ==========================================================
# MEDIAPIPE INIT
# ==========================================================
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_emotion(face_landmarks):
    # landmarks
    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    # mouth ratio
    mouth_width = abs(left_corner.x - right_corner.x)
    mouth_height = abs(upper_lip.y - lower_lip.y)
    ratio = mouth_height / mouth_width

    if ratio > 0.03:
        return "Happy", 90
    elif ratio < 0.015:
        return "Sad", 30
    else:
        return "Neutral", 60

def detect_eye_contact(face_landmarks):
    nose = face_landmarks.landmark[1]
    left_face = face_landmarks.landmark[234]
    right_face = face_landmarks.landmark[454]
    face_center = (left_face.x + right_face.x)/2
    deviation = abs(nose.x - face_center)
    if deviation < 0.03:
        return "Good Eye Contact", 95
    else:
        return "Looking Away", 35

def detect_posture(pose_landmarks):
    nose = pose_landmarks.landmark[0]
    left_shoulder = pose_landmarks.landmark[11]
    right_shoulder = pose_landmarks.landmark[12]
    shoulder_mid = (left_shoulder.x + right_shoulder.x)/2
    head_shift = abs(nose.x - shoulder_mid)
    if head_shift > 0.05:
        return "Slouching", 40
    else:
        return "Good Posture", 95

def calculate_confidence(emotion_score, eye_score, posture_score):
    return int(0.4*emotion_score + 0.3*eye_score + 0.3*posture_score)

def generate_feedback_from_confidence(confidence_score):
    if confidence_score > 80:
        return ["🎯 Excellent performance! Keep it up!"]
    elif confidence_score > 60:
        return ["👍 Good job, but room for improvement"]
    elif confidence_score > 40:
        return ["⚠️ Needs improvement, practice more"]
    else:
        return ["💪 Focus on posture, eye contact, and expression"]

# ==========================================================
# DASHBOARD LAYOUT
# ==========================================================
left_col, right_col = st.columns([2,1])
frame_placeholder = left_col.empty()

emotion_metric = right_col.empty()
eye_metric = right_col.empty()
posture_metric = right_col.empty()
confidence_metric = right_col.empty()

emotion_bar = right_col.progress(0)
eye_bar = right_col.progress(0)
posture_bar = right_col.progress(0)

# ==========================================================
# MAIN LOOP
# ==========================================================
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = face_mesh.process(rgb)
        pose_result = pose.process(rgb)

        emotion_text, emotion_score = "No Face", 0
        eye_text, eye_score = "No Face", 0
        posture_text, posture_score = "No Body", 0

        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                emotion_text, emotion_score = detect_emotion(face_landmarks)
                eye_text, eye_score = detect_eye_contact(face_landmarks)

        if pose_result.pose_landmarks:
            posture_text, posture_score = detect_posture(pose_result.pose_landmarks)

        confidence_score = calculate_confidence(emotion_score, eye_score, posture_score)

        # STORE HISTORY
        st.session_state.confidence_history.append(confidence_score)
        st.session_state.emotion_history.append(emotion_score)
        st.session_state.eye_history.append(eye_score)
        st.session_state.posture_history.append(posture_score)

        # UPDATE DASHBOARD
        emotion_metric.metric("Emotion", emotion_text)
        eye_metric.metric("Eye Contact", eye_text)
        posture_metric.metric("Posture", posture_text)
        confidence_metric.metric("Confidence Score", confidence_score)

        # progress bars
        emotion_bar.progress(emotion_score)
        eye_bar.progress(eye_score)
        posture_bar.progress(posture_score)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# ==========================================================
# FINAL REPORT
# ==========================================================
if st.session_state.interview_done and st.session_state.confidence_history:
    avg_confidence = int(sum(st.session_state.confidence_history)/len(st.session_state.confidence_history))
    avg_emotion = int(sum(st.session_state.emotion_history)/len(st.session_state.emotion_history))
    avg_eye = int(sum(st.session_state.eye_history)/len(st.session_state.eye_history))
    avg_posture = int(sum(st.session_state.posture_history)/len(st.session_state.posture_history))
    duration = int(time.time() - st.session_state.start_time)

    st.success("Interview Completed ✅")
    st.write("### 📊 Final Interview Report")
    st.write(f"⏱ Duration: {duration} seconds")
    st.write(f"📈 Average Confidence Score: {avg_confidence}")
    st.write(f"🙂 Emotion Score: {avg_emotion}")
    st.write(f"👀 Eye Contact Score: {avg_eye}")
    st.write(f"🪑 Posture Score: {avg_posture}")

    feedback_list = generate_feedback_from_confidence(avg_confidence)
    st.write("### 🤖 AI Feedback")
    for tip in feedback_list:
        st.write(tip)