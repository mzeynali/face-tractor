import cv2

from face_tractor import FaceTractor

"""
Change padding (in fact zooming in/out)
by changing normalized_distance parameter
"""
face_extractor = FaceTractor(normalized_distance=0.6)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    faces = face_extractor.normalized_faces(frame)
    if len(faces) > 0:
        face = faces[0]
        cv2.imshow("face", face.normalized_image)
    else:
        cv2.imshow("face", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
