
import csv
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import json


def check_av1_encoding(file_path):
    """
    Check if the video file uses AV1 encoding.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        video_info = json.loads(result.stdout)
        if "streams" in video_info and len(video_info["streams"]) > 0:
            codec_name = video_info["streams"][0].get("codec_name")
            return codec_name == "av1"
        return False
    except subprocess.CalledProcessError:
        return False


def download_and_cut_video(row, download_dir, output_parent_folder):
    """
    Download a video from the URL and cut it into segments based on time intervals.
    """
    # Extract the video URL and time interval
    video_url = row[1]
    time_intervals = eval(row[2])  # Convert the string to a list of intervals (e.g., [['0:00:00', '0:00:10']])

    video_id = video_url.replace("https://www.youtube.com/watch?v=", "")
    # Create a subfolder for each video
    sub_folder = os.path.join(download_dir, video_id)
    os.makedirs(sub_folder, exist_ok=True)

    # Temporary file for the full downloaded video
    temp_video_path = os.path.join(sub_folder, f"{video_id}_temp.mp4")
    target_path = os.path.join(output_parent_folder, f"{video_id}.mp4")

    # Download the video if not already downloaded
    if not os.path.exists(target_path):
        download_command = [
            "yt-dlp",
            "-U",
            "--output", temp_video_path,
            "--merge-output-format", "mp4",
            "--cookies", "/path/to/your/cookies.txt",  #change to your google cookies file
            "--no-check-certificate",
            "--geo-bypass",
            "--force-ipv4",
            video_url
        ]
        subprocess.run(download_command)

    # Check if the downloaded video is AV1 encoded
    if check_av1_encoding(temp_video_path):
        print(f"Skipping AV1 encoded video: {temp_video_path}")
        return

    # Process each time interval
    
    for start, end in time_intervals:
        segment_file_path = os.path.join(output_parent_folder, f"{video_id}.mp4")

        # Skip if the segment already exists
        if os.path.exists(segment_file_path):
            print(f"Segment {segment_file_path} already exists. Skipping...")
            continue

        # Cut the segment using ffmpeg with libx264 codec
        cut_command = [
            "ffmpeg",
            "-i", temp_video_path,
            "-ss", start,
            "-to", end,
            "-c:v", "libx264",
            segment_file_path
        ]
        subprocess.run(cut_command)

    # Optionally delete the temporary full video
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)


def process_csv(csv_file, download_dir, output_parent_folder, start_row=0, end_row=None):
    """
    Process the CSV file to download and cut videos based on rows.
    """
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        rows = [row for row in reader]

    # Slice the rows for the specified range
    if end_row is None:
        end_row = len(rows)
    rows = rows[start_row:end_row]

    # Use a thread pool to process rows in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda row: download_and_cut_video(row, download_dir, output_parent_folder), rows)


if __name__ == "__main__":
    #The complete video would be downloaded here
    download_dir = "/home/ubuntu/1065001-1070000"
    #The complete video would be clipped and move to here
    output_parent_folder = '/home/ubuntu/1065001-1070000_clip'
    # 创建输出目录（如果不存在）
    os.makedirs(output_parent_folder, exist_ok=True)
    # change the start row and end row
    process_csv('unbalanced_train_1000001-2000000.csv', download_dir, output_parent_folder,
    start_row=1065001-1000000,end_row=1070001-1000000)
