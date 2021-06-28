# coding:utf-8
from __future__ import print_function
import face_recognition
import pickle
import cv2
import time
import json
import io
import os
import re
import random
import shutil
import subprocess
from tqdm import tqdm

command_snap_shot="ffmpeg -ss {} -i {} -frames:v 1 {}"
#参数依次为:开始时间、输入视频文件名、截图名
command_video_cut="ffmpeg -ss {} -to {} -i {} -codec copy -avoid_negative_ts 1 {}"
#参数依次为:开始时间、持续时间、输入视频文件名、输出视频文件名
command_video_merge="ffmpeg -f concat -i {} -c copy {}"
#参数为:输出视频文件名
frames_every_capture=25
#每多少帧截取一次画面

folder_images="images"
#放置示例图片的文件夹
folder_data="data"
#示例图片采集的面孔放置的文件夹
folder_videos="videos"
#放置视频的文件夹
folder_output="output"
#输出文件夹
folder_recognition="recognition"
#识别的面孔文件夹


filename_video_input= "第1期-火箭少女101为篮球热血开唱，哈登清唱《荣誉星球》.mp4"
#需要cut的源视频，需要放置在folder_videos文件夹中
filename_video_output= "成品.mp4"
#cut完成后导出的视频名，将会放置在folder_videos文件夹中
filename_file_list= "filelist.txt"
#cut子视频目录文件
filename_static= "static.json"
#帧画面识别统计文件
filename_face_encodings="face_encodings.pkl"
#保存采集到的面孔的face_encoding的文件
filename_knn_model="trained_knn_model.clf"
#训练好的knn模型

resize_scale=0.5
#视频画面缩放因子
frames_to_recofnize_once=32
#每次批量识别的图片数量
threshold_duration = 10
#如果两个子cut间隔小于这个阈值，就合并为一个cut，单位:秒
threshold_recognition = 1
#如果所有比较中，tolerance为0.4的识别成功次数大于这个阈值，就认为此帧含有目标面孔
time_cut_pre=1
#前置缓冲时间，将会提前几秒开始剪
time_cut_post=1
#后置缓冲时间，将会推迟几秒结束剪

def generate_data_set():
    #从folder_images给出的资源中，截取样本面孔作为数据集
    filenames = [int(item.replace(".jpg", "")) for item in os.listdir(folder_data)]
    if len(filenames) == 0:
        label=0
    else:
        label=max(filenames)
    begin=time.time()
    for filename in os.listdir(folder_images):
        image=cv2.imread(os.path.join(folder_images,filename))
        if image is None:#有些图片会读不出来
            # print(filename)
            continue

        #防止图片过大，导致gpu显存溢出
        height, width = image.shape[:2]
        if height>width:
            long=height
        else:
            long=width
        scale = 1080 / float(long)
        image= cv2.resize(image, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_CUBIC)


        faces=face_recognition.face_locations(image,number_of_times_to_upsample=0,model="cnn")
        # faces = face_recognition.face_locations(image)
        # 如果你没有gpu,请使用cpu版↑，后同
        for (bottom,right,top,left) in faces:
            width=right-left
            height=top-bottom
            if width<50 or height<50:#太小的不要
                continue
            image_cut=image[bottom:top,left:right]
            label += 1
            filename_save="{}.jpg".format(label)
            path_save=os.path.join(folder_data, filename_save)
            cv2.imwrite(path_save,image_cut)
    end=time.time()
    print("生成数据集共计用时{}秒".format(round(float(end-begin),3)))

def generate_face_encodings():
    total_face_encoding = []
    for filename in os.listdir(folder_data):
        file_path=os.path.join(folder_data,filename)
        image=face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
        face_encoding_temp=face_recognition.face_encodings(image,face_locations,num_jitters=10)
        if len(face_encoding_temp)>0:
            total_face_encoding+=face_encoding_temp
            print(file_path)
    with open(filename_face_encodings,'wb') as fd:
        pickle.dump(total_face_encoding,fd)

def generate():
    generate_data_set()
    generate_face_encodings()

def face_dectect(filename_video_input):
    label=0
    total_face_encoding=pickle.load(open(filename_face_encodings,'rb'))
    folder_temp = filename_video_input.split(".")[0]
    folder_temp =os.path.join(folder_output,folder_temp)
    if not os.path.exists(folder_temp):
        os.mkdir(folder_temp)
    file_video=os.path.join(folder_videos,filename_video_input)
    video = cv2.VideoCapture(file_video)
    frames_total=video.get(cv2.CAP_PROP_FRAME_COUNT)
    progress=tqdm([i for i in range(int(frames_total))],desc="正在识别视频帧",unit='帧')
    frame_count = 1
    success = True
    begin = time.time()
    total={}
    frames=[]
    while success:
        success, frame = video.read()
        if (frame_count % frames_every_capture == 0):
            capture=int(frame_count/frames_every_capture)
            progress.update(frames_every_capture)
            frame = cv2.resize(frame, (0, 0),fx=resize_scale, fy=resize_scale,interpolation=cv2.INTER_CUBIC)
            # 视频画面可以缩小一些，但是阈值也要调整
            frames.append(frame)
            if len(frames)>=frames_to_recofnize_once:
                batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)
                for frame_index, face_locations in enumerate(batch_of_face_locations):
                    frame=frames[frame_index]
                    frame_number = capture - frames_to_recofnize_once + frame_index
                    face_encodings = face_recognition.face_encodings(frame, face_locations)
                    toleranses={
                        0.4:0,
                        0.5:0,
                        0.6:0
                    }
                    faces=[]
                    for (bottom, right, top, left), face_encoding in zip(face_locations, face_encodings):
                        toleranses_temp = {
                            0.4: 0,
                            0.5: 0,
                            0.6: 0
                        }
                        for toleranse in toleranses:
                            matches=face_recognition.compare_faces(total_face_encoding,face_encoding,tolerance=toleranse)
                            for match in matches:
                                if match:
                                    toleranses_temp[toleranse] += 1
                        for toleranse in toleranses_temp:
                            if toleranses_temp[toleranse]>toleranses[toleranse]:
                                toleranses[toleranse]=toleranses_temp[toleranse]
                        if toleranses_temp[0.4]>threshold_recognition:
                            face=[bottom,top,left,right]
                            faces.append(face)
                            image_cut = frame[bottom:top, left:right]
                            label += 1
                            filename_save = "{}-{}.jpg".format(label,frame_number)
                            path_file_save=os.path.join(folder_recognition,filename_save)
                            cv2.imwrite(path_file_save, image_cut)
                    total[frame_number]={
                        "toleranse":toleranses,
                        "faces":faces
                    }
                frames = []
        frame_count = frame_count + 1
        cv2.waitKey(1)
    end = time.time()
    progress.close()
    print("识别视频帧共计用时{}秒".format(round(float(end - begin), 3)))
    with io.open(os.path.join(folder_temp,filename_static),"w",encoding="utf-8") as fd:
        text = json.dumps(total, ensure_ascii=False, indent=4)
        fd.write(text)
    video.release()

def face_dectect_knn(filename_video_input):
    label=0
    knn_clf =pickle.load(open(filename_knn_model,'rb'))
    knn_clf.n_jobs=16
    folder_temp=filename_video_input.split(".")[0]
    folder_temp =os.path.join(folder_output,folder_temp)
    if not os.path.exists(folder_temp):
        os.mkdir(folder_temp)
    file_video=os.path.join(folder_videos,filename_video_input)
    video = cv2.VideoCapture(file_video)
    frames_total=video.get(cv2.CAP_PROP_FRAME_COUNT)
    progress=tqdm([i for i in range(int(frames_total))],desc="正在识别视频帧",unit='帧')
    frame_count = 1
    success = True
    begin = time.time()
    total={}
    frames=[]
    while success:
        success, frame = video.read()
        if (frame_count % frames_every_capture == 0):
            capture=int(frame_count/frames_every_capture)
            progress.update(frames_every_capture)
            frame = cv2.resize(frame, (0, 0),fx=resize_scale, fy=resize_scale,interpolation=cv2.INTER_CUBIC)
            # 视频画面可以缩小一些，但是阈值也要调整
            frames.append(frame)
            if len(frames)>=frames_to_recofnize_once:
                batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)
                frame_index_zip=[]
                face_encodings_zip=[]
                face_locations_zip=[]
                for frame_index, face_locations in enumerate(batch_of_face_locations):
                    if len(face_locations)==0:
                        continue
                    face_encodings = face_recognition.face_encodings(frame, face_locations)
                    for i in range(len(face_encodings)):
                        frame_index_zip.append(frame_index)
                    face_encodings_zip+=face_encodings
                    face_locations_zip+=face_locations
                predicts = knn_clf.predict(face_encodings_zip)
                for frame_index,(bottom, right, top, left), result in zip(frame_index_zip,face_locations_zip, predicts):
                    if result == "ycy":
                        frame = frames[frame_index]
                        frame_number = capture - frames_to_recofnize_once + frame_index
                        face=[bottom,top,left,right]
                        image_cut = frame[bottom:top, left:right]
                        label += 1
                        filename_save = "recognition/true/{}-{}.jpg".format(label,frame_number)
                        cv2.imwrite(filename_save, image_cut)
                        toleranses = {
                            0.4: 10,
                        }
                        if frame_number in total:
                            if "faces" in total[frame_number]:
                                total[frame_number]["faces"].append(face)
                            else:
                                total[frame_number]["faces"]=[face]
                        else:
                            total[frame_number]["toleranse"] = toleranses
                    else:
                        frame = frames[frame_index]
                        frame_number = capture - frames_to_recofnize_once + frame_index
                        image_cut = frame[bottom:top, left:right]
                        label += 1
                        filename_save = "recognition/false/{}-{}.jpg".format(label,frame_number)
                        cv2.imwrite(filename_save, image_cut)
                frames = []
        frame_count = frame_count + 1
        cv2.waitKey(1)
    end = time.time()
    progress.close()
    print("识别视频帧共计用时{}秒".format(round(float(end - begin), 3)))
    with io.open(os.path.join(folder_temp,filename_static),"w",encoding="utf-8") as fd:
        text = json.dumps(total, ensure_ascii=False, indent=4)
        fd.write(text)
    video.release()

def analyse(filename_video_input):
    def convert_to_time(second):
        hour=int(second/3600)
        minute=int(second%3600/60)
        second=second%60
        result="%02d:%02d:%02d"%(hour,minute,second)
        return result
    images_recognition=os.listdir(folder_recognition)
    cut_points=[]
    folder_temp=filename_video_input.split(".")[0]
    folder_temp =os.path.join(folder_output,folder_temp)
    file_static=os.path.join(folder_temp,filename_static)
    data=json.loads(open(file_static).read())
    for capture in data:
        count_4=data[capture]["toleranse"]["0.4"]
        if count_4>threshold_recognition:
            cut_points.append(int(capture))
    group=[]
    begin=cut_points[0]
    for i in range(len(cut_points)-1):
        now=cut_points[i]
        next=cut_points[i+1]
        end = now
        if next>now+threshold_duration:
            group.append((begin,end))
            begin=next

    file_video=os.path.join(folder_videos,filename_video_input)
    video = cv2.VideoCapture(file_video)
    fps=video.get(cv2.CAP_PROP_FPS)
    label=0
    for begin,end in group:
        start =begin*frames_every_capture//fps-time_cut_pre
        if start<0:
            start=0
        terminal=end*frames_every_capture//fps+time_cut_post
        start=int(start)
        terminal=int(terminal)
        label+=1
        filename="cut{}.mp4".format(label)
        folder_cut_images=os.path.join(folder_temp,"cut{}".format(label))
        if not os.path.exists(folder_cut_images):
            os.mkdir(folder_cut_images)
        for image in images_recognition:
            if not ".jpg" in image:
                continue
            second=image.split("-")[1].replace(".jpg","")
            second=int(second)*frames_every_capture//fps
            if second>=start and second<=terminal:
                src=os.path.join(folder_recognition,image)
                desc=os.path.join(folder_cut_images,image)
                shutil.move(src,desc)
        file_output=os.path.join(folder_temp,filename)
        if not os.path.exists(file_output):
            filename_input=os.path.join(folder_videos, filename_video_input)
            command=command_video_cut.format(convert_to_time(start), convert_to_time(terminal), "\"{}\"".format(filename_input), "\"{}\"".format(file_output))
            print(command)
            shell = subprocess.Popen(command)
            shell.wait()

def merge():
    folder_temp = filename_video_input.split(".")[0]
    folder_temp = os.path.join(folder_output, folder_temp)
    file_lists=open(os.path.join(folder_temp,filename_file_list), "w")
    video_list=[]
    for filename in os.listdir(folder_temp):
        if not ".mp4" in filename:
            result=re.findall("(?<=cut).*",filename)
            if len(result)>0:
                video_list.append(int(result[0]))
    for filename in os.listdir(folder_temp):
        if ".mp4" in filename:
            result=re.findall("(?<=cut).+?(?=\.mp4)",filename)
            if len(result)>0:
                if not int(result[0]) in video_list:
                    os.remove(os.path.join(folder_temp,filename))
    video_list.sort()
    for number in video_list:
        file_lists.write("file \'cut{}.mp4\'\n".format(number))
    file_lists.close()
    command=command_video_merge.format(filename_file_list, filename_video_output)
    print(command)
    process= subprocess.Popen(command,cwd=folder_temp)
    process.wait()

def init():
    if not os.path.exists(folder_videos):
        os.mkdir(folder_videos)
    if not os.path.exists(folder_output):
        os.mkdir(folder_output)
    if not os.path.exists(folder_data):
        os.mkdir(folder_data)
    if not os.path.exists(folder_images):
        os.mkdir(folder_images)

def main():
    init()
    # generate_face_encodings()
    # exit(0)
    if not os.path.exists(filename_face_encodings):
        #如果face_encoding文件不存在，就进行采集
        generate()
    # face_dectect_knn(filename_video_input)
    begin = time.time()
    face_dectect(filename_video_input)
    analyse(filename_video_input)
    merge()
    end=time.time()
    print("共计用时{}秒".format(round(float(end - begin), 3)))

if __name__=="__main__":
    main()