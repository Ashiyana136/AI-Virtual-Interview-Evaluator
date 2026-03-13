import streamlit as st
import cv2
import mediapipe as mp
import math
import time

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="AI Virtual Interview Evaluator", layout="wide")
st.title("🧠 AI Virtual Interview Coach Dashboard")

# ==========================================================
# SESSION STATE
# ==========================================================
if "run" not in st.session_state:
    st.session_state.run = False

if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = []

if "interview_done" not in st.session_state:
    st.session_state.interview_done = False

if "start_time" not in st.session_state:
    st.session_state.start_time = 0

# ==========================================================
# START / STOP FUNCTIONS
# ==========================================================
def start():
    st.session_state.run = True
    st.session_state.interview_done = False
    st.session_state.confidence_history = []
    st.session_state.start_time = time.time()

def stop():
    st.session_state.run = False
    st.session_state.interview_done = True

col1, col2 = st.columns(2)
col1.button("🎥 Start Interview", on_click=start)
col2.button("🛑 Stop Interview", on_click=stop)

# ==========================================================
# MEDIAPIPE INITIALIZATION
# ==========================================================
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================================
# HELPER FUNCTION
# ==========================================================
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# ==========================================================
# EMOTION DETECTION
# ==========================================================
def detect_emotion(face_landmarks):

    # Key landmarks
    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]

    # Face width normalization
    face_width = abs(left_cheek.x - right_cheek.x)

    # Mouth width & height
    mouth_width = abs(left_corner.x - right_corner.x)
    mouth_height = abs(upper_lip.y - lower_lip.y)

    # Normalize values
    mouth_ratio = mouth_width / face_width
    openness_ratio = mouth_height / face_width

    # Smile curve (corners vs lip center)
    smile_curve = ((left_corner.y + right_corner.y) / 2) - upper_lip.y


    # Happy → wide mouth + slight curve
    if mouth_ratio > 0.38 and smile_curve < -0.005:
        return 90, "Happy"


    # Neutral → medium ratio
    elif 0.30 < mouth_ratio <= 0.38:
        return 60, "Neutral"

    # Sad → corners down
    elif smile_curve > 0.01:
        return 35, "Sad"

    else:
        return 55, "Neutral"

# ==========================================================
# EYE CONTACT
# ==========================================================
def detect_eye_contact(face_landmarks):
    nose = face_landmarks.landmark[1]
    left_face = face_landmarks.landmark[234]
    right_face = face_landmarks.landmark[454]

    face_center = (left_face.x + right_face.x) / 2
    deviation = abs(nose.x - face_center)

    if deviation < 0.025:
        return 95, "Good Eye Contact"
    else:
        return 35, "Looking Away"

# ==========================================================
# POSTURE DETECTION
# ==========================================================
def detect_posture(pose_landmarks):
    left_shoulder = pose_landmarks.landmark[11]
    right_shoulder = pose_landmarks.landmark[12]
    nose = pose_landmarks.landmark[0]

    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    head_shift = abs(nose.x - shoulder_mid_x)

    if shoulder_diff < 0.035 and head_shift < 0.04:
        return 95, "Good Posture"
    else:
        return 40, "Bad Posture"

# ==========================================================
# CONFIDENCE SCORE
# ==========================================================
def calculate_confidence(emotion, eye, posture):
    score = (0.4 * emotion) + (0.3 * eye) + (0.3 * posture)
    return int(score)

# ==========================================================
# DASHBOARD LAYOUT
# ==========================================================
left_col, right_col = st.columns([2, 1])

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

        emotion_score = 0
        eye_score = 0
        posture_score = 0

        emotion_text = "No Face"
        eye_text = "No Face"
        posture_text = "No Body"

        # FACE
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                emotion_score, emotion_text = detect_emotion(face_landmarks)
                eye_score, eye_text = detect_eye_contact(face_landmarks)

        # POSE
        if pose_result.pose_landmarks:
            posture_score, posture_text = detect_posture(pose_result.pose_landmarks)

        # CONFIDENCE
        confidence_score = calculate_confidence(
            emotion_score, eye_score, posture_score
        )

        st.session_state.confidence_history.append(confidence_score)

        # DASHBOARD
        emotion_metric.metric("Emotion", emotion_text)
        eye_metric.metric("Eye Contact", eye_text)
        posture_metric.metric("Posture", posture_text)
        confidence_metric.metric("Confidence Score", f"{confidence_score}")

        emotion_bar.progress(emotion_score)
        eye_bar.progress(eye_score)
        posture_bar.progress(posture_score)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# ==========================================================
# FINAL REPORT
# ==========================================================
if st.session_state.interview_done:

    if st.session_state.confidence_history:

        avg_confidence = sum(st.session_state.confidence_history) / len(
            st.session_state.confidence_history
        )
        duration = int(time.time() - st.session_state.start_time)

        st.success("Interview Completed ✅")
        st.write("### Final Report")
        st.write(f"⏱ Duration: {duration} seconds")
        st.write(f"📊 Average Confidence: {int(avg_confidence)}%")

        if avg_confidence > 75:
            st.success("Excellent performance! 🎯")
        elif avg_confidence > 50:
            st.warning("Good, but room for improvement 👍")
        else:
            st.error("Needs improvement. Practice more 💪")

# import streamlit as st
# import cv2
# import mediapipe as mp
# import math
# import time

# # ==========================================================
# # PAGE CONFIGURATION
# # ==========================================================

# st.set_page_config(page_title="AI Virtual Interview Evaluator", layout="wide")
# st.title("🧠 AI Virtual Interview Coach Dashboard")

# # ==========================================================
# # SESSION STATE VARIABLES
# # ==========================================================

# if "run" not in st.session_state:
#     st.session_state.run = False

# if "confidence_history" not in st.session_state:
#     st.session_state.confidence_history = []

# if "interview_done" not in st.session_state:
#     st.session_state.interview_done = False

# if "start_time" not in st.session_state:
#     st.session_state.start_time = 0

# # ==========================================================
# # BUTTON FUNCTIONS
# # ==========================================================

# def start_interview():
#     st.session_state.run = True
#     st.session_state.interview_done = False
#     st.session_state.confidence_history = []
#     st.session_state.start_time = time.time()

# def stop_interview():
#     st.session_state.run = False
#     st.session_state.interview_done = True

# col1, col2 = st.columns(2)

# col1.button("🎥 Start Interview", on_click=start_interview)
# col2.button("🛑 Stop Interview", on_click=stop_interview)

# # ==========================================================
# # MEDIAPIPE INITIALIZATION
# # ==========================================================

# mp_face_mesh = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # ==========================================================
# # HELPER FUNCTION
# # ==========================================================

# def distance(p1, p2):
#     return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# # ==========================================================
# # EMOTION DETECTION
# # ==========================================================

# def detect_emotion(face_landmarks):

#     left_corner = face_landmarks.landmark[61]
#     right_corner = face_landmarks.landmark[291]
#     upper_lip = face_landmarks.landmark[13]
#     lower_lip = face_landmarks.landmark[14]
#     left_cheek = face_landmarks.landmark[234]
#     right_cheek = face_landmarks.landmark[454]

#     face_width = abs(left_cheek.x - right_cheek.x)

#     mouth_width = abs(left_corner.x - right_corner.x)
#     mouth_height = abs(upper_lip.y - lower_lip.y)

#     mouth_ratio = mouth_width / face_width
#     smile_curve = ((left_corner.y + right_corner.y) / 2) - upper_lip.y

#     if mouth_ratio > 0.38 and smile_curve < -0.005:
#         return 90, "Happy"

#     elif 0.30 < mouth_ratio <= 0.38:
#         return 60, "Neutral"

#     elif smile_curve > 0.01:
#         return 35, "Sad"

#     else:
#         return 55, "Neutral"

# # ==========================================================
# # EYE CONTACT DETECTION
# # ==========================================================

# def detect_eye_contact(face_landmarks):

#     nose = face_landmarks.landmark[1]
#     left_face = face_landmarks.landmark[234]
#     right_face = face_landmarks.landmark[454]

#     face_center = (left_face.x + right_face.x) / 2
#     deviation = abs(nose.x - face_center)

#     if deviation < 0.025:
#         return 95, "Good Eye Contact"
#     else:
#         return 35, "Looking Away"

# # ==========================================================
# # POSTURE DETECTION
# # ==========================================================

# def detect_posture(pose_landmarks):

#     left_shoulder = pose_landmarks.landmark[11]
#     right_shoulder = pose_landmarks.landmark[12]
#     nose = pose_landmarks.landmark[0]

#     shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
#     shoulder_mid = (left_shoulder.x + right_shoulder.x) / 2
#     head_shift = abs(nose.x - shoulder_mid)

#     if shoulder_diff < 0.035 and head_shift < 0.04:
#         return 95, "Good Posture"
#     else:
#         return 40, "Bad Posture"

# # ==========================================================
# # CONFIDENCE SCORE
# # ==========================================================

# def calculate_confidence(emotion, eye, posture):

#     score = (0.4 * emotion) + (0.3 * eye) + (0.3 * posture)
#     return int(score)

# # ==========================================================
# # FEEDBACK SYSTEM
# # ==========================================================

# def generate_feedback(emotion, eye, posture):

#     feedback = []

#     if emotion < 50:
#         feedback.append("🙂 Try to maintain a positive facial expression")

#     if eye < 60:
#         feedback.append("👀 Maintain better eye contact with the camera")

#     if posture < 60:
#         feedback.append("🪑 Sit straight and keep your shoulders aligned")

#     if not feedback:
#         feedback.append("🎯 Excellent interview performance!")

#     return feedback

# # ==========================================================
# # DASHBOARD LAYOUT
# # ==========================================================

# left_col, right_col = st.columns([2,1])

# frame_placeholder = left_col.empty()

# emotion_metric = right_col.empty()
# eye_metric = right_col.empty()
# posture_metric = right_col.empty()
# confidence_metric = right_col.empty()

# emotion_bar = right_col.progress(0)
# eye_bar = right_col.progress(0)
# posture_bar = right_col.progress(0)

# confidence_chart = right_col.empty()

# # ==========================================================
# # MAIN INTERVIEW LOOP
# # ==========================================================

# if st.session_state.run:

#     cap = cv2.VideoCapture(0)

#     while st.session_state.run and cap.isOpened():

#         success, frame = cap.read()
#         if not success:
#             break

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         face_result = face_mesh.process(rgb)
#         pose_result = pose.process(rgb)

#         emotion_score = 0
#         eye_score = 0
#         posture_score = 0

#         emotion_text = "No Face"
#         eye_text = "No Face"
#         posture_text = "No Body"

#         # FACE ANALYSIS
#         if face_result.multi_face_landmarks:
#             for face_landmarks in face_result.multi_face_landmarks:

#                 emotion_score, emotion_text = detect_emotion(face_landmarks)
#                 eye_score, eye_text = detect_eye_contact(face_landmarks)

#         # POSTURE ANALYSIS
#         if pose_result.pose_landmarks:

#             posture_score, posture_text = detect_posture(pose_result.pose_landmarks)

#         # CONFIDENCE SCORE
#         confidence_score = calculate_confidence(
#             emotion_score,
#             eye_score,
#             posture_score
#         )

#         st.session_state.confidence_history.append(confidence_score)

#         # DASHBOARD METRICS
#         emotion_metric.metric("Emotion", f"{emotion_text} ({emotion_score})")

#         eye_metric.metric("Eye Contact", f"{eye_text} ({eye_score})")

#         posture_metric.metric("Posture", f"{posture_text} ({posture_score})")

#         confidence_metric.metric("Confidence Score", f"{confidence_score} / 100")

#         # PROGRESS BARS
#         emotion_bar.progress(emotion_score)
#         eye_bar.progress(eye_score)
#         posture_bar.progress(posture_score)

#         # CONFIDENCE GRAPH
#         confidence_chart.line_chart(st.session_state.confidence_history)

#         # VIDEO FRAME
#         frame_placeholder.image(frame, channels="BGR")

#     cap.release()
#     cv2.destroyAllWindows()

# # ==========================================================
# # FINAL REPORT
# # ==========================================================

# if st.session_state.interview_done:

#     if st.session_state.confidence_history:

#         avg_confidence = sum(st.session_state.confidence_history) / len(
#             st.session_state.confidence_history
#         )

#         duration = int(time.time() - st.session_state.start_time)

#         st.success("Interview Completed ✅")

#         st.write("### 📊 Final Interview Report")

#         st.write(f"⏱ Duration: {duration} seconds")

#         st.write(f"📈 Average Confidence Score: {int(avg_confidence)}%")

#         feedback_list = generate_feedback(
#             avg_confidence,
#             avg_confidence,
#             avg_confidence
#         )

#         st.write("### 🤖 AI Feedback")

#         for tip in feedback_list:
#             st.write(tip)

#         if avg_confidence > 75:
#             st.success("Excellent performance! 🎯")

#         elif avg_confidence > 50:
#             st.warning("Good, but room for improvement 👍")

#         else:
#             st.error("Needs improvement. Practice more 💪")

# import streamlit as st
# import cv2
# import mediapipe as mp
# import math
# import time

# # ==========================================================
# # PAGE CONFIGURATION
# # ==========================================================

# st.set_page_config(page_title="AI Virtual Interview Evaluator", layout="wide")
# st.title("🧠 AI Virtual Interview Coach Dashboard")

# # ==========================================================
# # SESSION STATE VARIABLES
# # ==========================================================

# if "run" not in st.session_state:
#     st.session_state.run = False

# if "confidence_history" not in st.session_state:
#     st.session_state.confidence_history = []

# if "emotion_history" not in st.session_state:
#     st.session_state.emotion_history = []

# if "eye_history" not in st.session_state:
#     st.session_state.eye_history = []

# if "posture_history" not in st.session_state:
#     st.session_state.posture_history = []

# if "interview_done" not in st.session_state:
#     st.session_state.interview_done = False

# if "start_time" not in st.session_state:
#     st.session_state.start_time = 0

# # ==========================================================
# # BUTTON FUNCTIONS
# # ==========================================================

# def start_interview():
#     st.session_state.run = True
#     st.session_state.interview_done = False

#     st.session_state.confidence_history = []
#     st.session_state.emotion_history = []
#     st.session_state.eye_history = []
#     st.session_state.posture_history = []

#     st.session_state.start_time = time.time()


# def stop_interview():
#     st.session_state.run = False
#     st.session_state.interview_done = True


# col1, col2 = st.columns(2)
# col1.button("🎥 Start Interview", on_click=start_interview)
# col2.button("🛑 Stop Interview", on_click=stop_interview)

# # ==========================================================
# # MEDIAPIPE INITIALIZATION
# # ==========================================================

# mp_face_mesh = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose

# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # ==========================================================
# # HELPER FUNCTION
# # ==========================================================

# def distance(p1, p2):
#     return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# # ==========================================================
# # EMOTION DETECTION
# # ==========================================================

# def detect_emotion(face_landmarks):

#     left_corner = face_landmarks.landmark[61]
#     right_corner = face_landmarks.landmark[291]
#     upper_lip = face_landmarks.landmark[13]
#     lower_lip = face_landmarks.landmark[14]
#     left_cheek = face_landmarks.landmark[234]
#     right_cheek = face_landmarks.landmark[454]

#     face_width = abs(left_cheek.x - right_cheek.x)

#     mouth_width = abs(left_corner.x - right_corner.x)
#     mouth_height = abs(upper_lip.y - lower_lip.y)

#     mouth_ratio = mouth_width / face_width
#     smile_curve = ((left_corner.y + right_corner.y) / 2) - upper_lip.y

#     if mouth_ratio > 0.38 and smile_curve < -0.005:
#         return 90, "Happy"

#     elif 0.30 < mouth_ratio <= 0.38:
#         return 60, "Neutral"

#     elif smile_curve > 0.01:
#         return 35, "Sad"

#     else:
#         return 55, "Neutral"

# # ==========================================================
# # EYE CONTACT DETECTION
# # ==========================================================

# def detect_eye_contact(face_landmarks):

#     nose = face_landmarks.landmark[1]
#     left_face = face_landmarks.landmark[234]
#     right_face = face_landmarks.landmark[454]

#     face_center = (left_face.x + right_face.x) / 2
#     deviation = abs(nose.x - face_center)

#     if deviation < 0.025:
#         return 95, "Good Eye Contact"
#     else:
#         return 35, "Looking Away"

# # ==========================================================
# # POSTURE DETECTION
# # ==========================================================

# def detect_posture(pose_landmarks):

#     left_shoulder = pose_landmarks.landmark[11]
#     right_shoulder = pose_landmarks.landmark[12]
#     nose = pose_landmarks.landmark[0]

#     shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
#     shoulder_mid = (left_shoulder.x + right_shoulder.x) / 2
#     head_shift = abs(nose.x - shoulder_mid)

#     if shoulder_diff < 0.035 and head_shift < 0.04:
#         return 95, "Good Posture"
#     else:
#         return 40, "Bad Posture"

# # ==========================================================
# # CONFIDENCE SCORE
# # ==========================================================

# def calculate_confidence(emotion, eye, posture):

#     score = (0.4 * emotion) + (0.3 * eye) + (0.3 * posture)
#     return int(score)

# # ==========================================================
# # FEEDBACK SYSTEM
# # ==========================================================

# def generate_feedback(emotion, eye, posture):

#     feedback = []

#     if emotion < 50:
#         feedback.append("🙂 Try to maintain a positive facial expression")

#     if eye < 60:
#         feedback.append("👀 Maintain better eye contact with the camera")

#     if posture < 60:
#         feedback.append("🪑 Sit straight and keep your shoulders aligned")

#     if not feedback:
#         feedback.append("🎯 Excellent interview performance!")

#     return feedback

# # ==========================================================
# # DASHBOARD LAYOUT
# # ==========================================================

# left_col, right_col = st.columns([2,1])

# frame_placeholder = left_col.empty()

# emotion_metric = right_col.empty()
# eye_metric = right_col.empty()
# posture_metric = right_col.empty()
# confidence_metric = right_col.empty()

# emotion_bar = right_col.progress(0)
# eye_bar = right_col.progress(0)
# posture_bar = right_col.progress(0)

# confidence_chart = right_col.empty()

# # ==========================================================
# # MAIN INTERVIEW LOOP
# # ==========================================================

# if st.session_state.run:

#     cap = cv2.VideoCapture(0)

#     while st.session_state.run and cap.isOpened():

#         success, frame = cap.read()
#         if not success:
#             break

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         face_result = face_mesh.process(rgb)
#         pose_result = pose.process(rgb)

#         emotion_score = 0
#         eye_score = 0
#         posture_score = 0

#         emotion_text = "No Face"
#         eye_text = "No Face"
#         posture_text = "No Body"

#         # FACE ANALYSIS
#         if face_result.multi_face_landmarks:

#             for face_landmarks in face_result.multi_face_landmarks:

#                 emotion_score, emotion_text = detect_emotion(face_landmarks)
#                 eye_score, eye_text = detect_eye_contact(face_landmarks)

#         # POSTURE ANALYSIS
#         if pose_result.pose_landmarks:

#             posture_score, posture_text = detect_posture(
#                 pose_result.pose_landmarks
#             )

#         # CONFIDENCE SCORE
#         confidence_score = calculate_confidence(
#             emotion_score,
#             eye_score,
#             posture_score
#         )

#         # STORE HISTORY
#         st.session_state.confidence_history.append(confidence_score)
#         st.session_state.emotion_history.append(emotion_score)
#         st.session_state.eye_history.append(eye_score)
#         st.session_state.posture_history.append(posture_score)

#         # DASHBOARD METRICS
#         emotion_metric.metric("Emotion", f"{emotion_text} ({emotion_score})")

#         eye_metric.metric("Eye Contact", f"{eye_text} ({eye_score})")

#         posture_metric.metric("Posture", f"{posture_text} ({posture_score})")

#         confidence_metric.metric("Confidence Score", f"{confidence_score}/100")

#         # PROGRESS BARS
#         emotion_bar.progress(emotion_score)
#         eye_bar.progress(eye_score)
#         posture_bar.progress(posture_score)

#         # CONFIDENCE GRAPH
#         confidence_chart.line_chart(st.session_state.confidence_history)

#         # VIDEO DISPLAY
#         frame_placeholder.image(frame, channels="BGR")

#     cap.release()
#     cv2.destroyAllWindows()

# # ==========================================================
# # FINAL REPORT
# # ==========================================================

# if st.session_state.interview_done:

#     if st.session_state.confidence_history:

#         avg_confidence = sum(st.session_state.confidence_history) / len(
#             st.session_state.confidence_history
#         )

#         avg_emotion = sum(st.session_state.emotion_history) / len(
#             st.session_state.emotion_history
#         )

#         avg_eye = sum(st.session_state.eye_history) / len(
#             st.session_state.eye_history
#         )

#         avg_posture = sum(st.session_state.posture_history) / len(
#             st.session_state.posture_history
#         )

#         duration = int(time.time() - st.session_state.start_time)

#         st.success("Interview Completed ✅")

#         st.write("### 📊 Final Interview Report")

#         st.write(f"⏱ Duration: {duration} seconds")

#         st.write(f"📈 Average Confidence Score: {int(avg_confidence)}%")

#         st.write(f"🙂 Emotion Score: {int(avg_emotion)}")

#         st.write(f"👀 Eye Contact Score: {int(avg_eye)}")

#         st.write(f"🪑 Posture Score: {int(avg_posture)}")

#         feedback_list = generate_feedback(
#             avg_emotion,
#             avg_eye,
#             avg_posture
#         )

#         st.write("### 🤖 AI Feedback")

#         for tip in feedback_list:
#             st.write(tip)

#         if avg_confidence > 75:
#             st.success("Excellent performance! 🎯")

#         elif avg_confidence > 50:
#             st.warning("Good, but room for improvement 👍")

#         else:
#             st.error("Needs improvement. Practice more 💪")