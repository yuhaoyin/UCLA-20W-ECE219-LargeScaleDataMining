The presented dataset is composed of two tsv files named "youtube_videos.tsv" 
and "transcoding_mesurment.tsv". The first contains 10 columns of fundamental 
video characteristics for 1.6 million youtube videos; It contains YouTube video id, 
duration, bitrate(total in Kbits), bitrate(video bitrate in Kbits), 
height(in pixle), width(in pixles), framrate, estimated framerate, codec, 
category, and direct video link. This dataset can be used to gain insight
in characteristics of consumer videos found on UGC(Youtube).

The second file of our dataset contains 20 columns(see column names for names) 
which include input and output video characteristics along with their transcoding 
time and memory resource requirements while transcoding videos to diffrent but 
valid formats. The second dataset was collected based on experiments on an Intel 
i7-3720QM CPU through randomly picking two rows from the first dataset and using 
these as input and output parameters of a video transcoding application, ffmpeg 4 . 
In section 6 we will use the second dataset to build a transcoding time prediction
model and show the significance of our datasets.

For more information please read the associated paper.

