import cv2

count = 0
cv2.waitKey(4000)
frameDirectory = f"/Users/mehulmathur/Desktop/Main Folder 0/Python Files/computer_vision_workshop/Final Project Folder/frames/FinalFrames0(1)"

while True:
   image = cv2.imread(f"{frameDirectory}/frame_{count}.jpg")
   cv2.imshow(f"frame_{count}", image)
   if cv2.waitKey(1) & 0xFF==ord('q'):
      break
   cv2.destroyAllWindows()
   count+=1