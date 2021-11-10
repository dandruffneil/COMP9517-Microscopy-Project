## Task 1:Segment and Track Cells



To segment the cells, we first using median blur and gamma correction to enhance the original image. Then use ostu binarization in opencv for segmentation. Finally we find coutours and centers of each cell.

For tracking, we use Kalman filter to track each cell and assign unique id to each cell among different frames.



# References -
https://github.com/mabhisharma/Multi-Object-Tracking-with-Kalman-Filter
