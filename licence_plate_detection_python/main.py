# Libraries which required.
import cv2
from ultralytics import YOLO
import easyocr

# Load the custom YOLOv8 model which we trainied for the licence plate detection 
model_path = 'best.pt'  # Adjust to your model's actual path
model = YOLO(model_path)

# need to  Initialize EasyOCR reader for the text extraction after licence plate detected.
reader = easyocr.Reader(['en'])  # Specify the language; 'en' for English

# Function to perform inference on a video
def detect_license_plate_in_video(video_path, output_file):
    # Open the video file   # put 0 for the real time analysis
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    with open(output_file, 'a') as f:  # Open the output file in append mode
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if there are no more frames

            # Run inference on the frame
            results = model(frame)

            # Process results
            for result in results:  # Iterate over results
                detections = result.boxes  # Access the boxes for this frame
                for box in detections:
                    # Each box contains [x1, y1, x2, y2, confidence, class]
                    x1, y1, x2, y2 = map(int, box.xyxy[0][:4])  # Get box coordinates
                    conf = box.conf[0]  # Get confidence score
                    
                    # Crop the detected license plate area
                    license_plate_img = frame[y1:y2, x1:x2]

                    # Use EasyOCR to extract text from the cropped license plate
                    license_plate_result = reader.readtext(license_plate_img)

                    # Extract and format the text from the result
                    license_plate_text = " ".join([text[1] for text in license_plate_result]).strip()

                    if license_plate_text:  # Check if text is detected
                        # Write the license plate text to the output file
                        f.write(license_plate_text + '\n')

                    # Draw bounding box and text on the frame so you see text on the screen
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{license_plate_text}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)

            # Resize the frame to 480x320
            resized_frame = cv2.resize(frame, (480, 320))

            # Show the frame with detections
            cv2.imshow('Detections', resized_frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'demo.mp4'  # Replace with the path to your video file
output_file = 'license_plates.txt'  # Output file for license plate texts
detect_license_plate_in_video(video_path, output_file)
