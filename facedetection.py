import cv2
import numpy as np
import tensorflow as tf

# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Load CNN model for face verification
def load_cnn_model():
    model = tf.keras.models.load_model('face_verification_cnn_model.h5')
    return model

# Detect faces using YOLO
def detect_faces(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    detected_faces = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        detected_faces.append(box)
    
    return detected_faces

# Verify face using CNN
def verify_face(face, model):
    face = cv2.resize(face, (128, 128))  # Resize to the input size expected by the CNN
    face = face.astype('float32') / 255.0  # Normalize
    face = np.expand_dims(face, axis=0)
    prediction = model.predict(face)
    return prediction[0][0]  # Return the similarity score

# Main function
def main():
    video_capture = cv2.VideoCapture(0)
    yolo_net, yolo_output_layers = load_yolo_model()
    cnn_model = load_cnn_model()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        faces = detect_faces(frame, yolo_net, yolo_output_layers)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            similarity_score = verify_face(face, cnn_model)
            
            label = 'Verified' if similarity_score > 0.5 else 'Not Verified'
            color = (0, 255, 0) if label == 'Verified' else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('Face Verification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
