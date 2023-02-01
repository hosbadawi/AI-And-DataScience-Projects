##################################################################################################### Libraries
# Import the required Libraries
from __future__ import unicode_literals
from pytube import YouTube
import os
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import random
import pyautogui
from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mymovie
import requests
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from pyvirtualdisplay import Display
import concurrent.futures
from tqdm import tqdm
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

##################################################################################################### Functions
class YoutubeScrapping:

    @staticmethod
    def remove_vid1(): # it will remove the video that has name 1.mp4 (this helps the automatic upload function)
        folder = os.getcwd() + "\\Videos\\"
        path = folder + "1.mp4"
        os.remove(path)

    @staticmethod
    def rename_videos(): # it will remane all the downloaded videos to -->> 1.mp4, 2.mp4, 3.mp4 ...etc
        folder = os.getcwd() + "\\Videos\\"
        count = 1
        try:
            for file_name in os.listdir(folder):
                source = folder + file_name
                destination = folder + str(count) + ".mp4"
                os.rename(source, destination)
                count += 1
        except:
            print("Files already renamed")

    @staticmethod
    def rename_audios(): # it will remane all the downloaded Audios to -->> 1.mp3, 2.mp3, 3.mp3 ...etc
        folder = os.getcwd() + "\\Audios\\"
        count = 1
        try:
            for file_name in os.listdir(folder):
                source = folder + file_name
                destination = folder + str(count) + ".mp3"
                os.rename(source, destination)
                count += 1
        except:
            print("Files already renamed")

    @staticmethod
    def mkdir(): # it will create two folders (Videos, and Audios) in the project files
        directory0 = "Videos"
        directory1 = "Audios"
        parent_dir = os.getcwd()
        path0 = os.path.join(parent_dir, directory0)
        path1 = os.path.join(parent_dir, directory1)
        try:
            os.mkdir(path0)
            os.mkdir(path1)
        except:
            print("Folders already exist.")

    @staticmethod
    def unique(mylist):  # get the unique urls from the list
        unique_list = []
        for i in mylist:
            if i not in unique_list:
                unique_list.append(i)
        return unique_list

    @staticmethod
    def create_driver():  # create the driver
        option = webdriver.ChromeOptions()
        # option.binary_location = "C:/Program Files/Google/Chrome Beta/Application/chrome.exe"
        # option.add_argument("window-size=1400,600")
        option.add_experimental_option("useAutomationExtension", False)
        option.add_experimental_option("excludeSwitches", ["enable-automation"])

        # avoiding detection
        option.add_argument('--disable-blink-features=AutomationControlled')
        option.add_argument("disable-infobars");
        option.add_argument("--start-maximized");
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'

        option.add_argument("--profile-directory=Default")
        option.add_argument("--user-data-dir=C:\\Users\\mm\\AppData\\Local\\Google\\Chrome\\User Data")

        option.add_argument('user-agent={0}'.format(user_agent))
        ua = UserAgent()
        header = ua.chrome
        option.add_argument(f'user-agent={header}')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)
        return driver

    @staticmethod
    def video_from_link(link):  # download video from its link
        start = time.time()
        yt = YouTube(link)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        out_file = video.download(output_path="./Videos")
        base, ext = os.path.splitext(out_file)
        random_number = random.randint(0, 999999999)
        new_file = base + str(random_number) + '.mp4'
        os.rename(out_file, new_file)
        stop = time.time()
        Time = stop - start
        return print("Done In ->> ", Time, "s")

    @staticmethod
    def audio_from_link(link):  # download audio from its link
        start = time.time()
        yt = YouTube(link)
        audio = yt.streams.filter(only_audio=True).first()
        out_file = audio.download(output_path="./Audios")
        base, ext = os.path.splitext(out_file)
        random_number = random.randint(0, 999999999)
        new_file = base + str(random_number) + '.mp3'
        os.rename(out_file, new_file)
        stop = time.time()
        Time = stop - start
        return print("Done In ->> ", Time, "s")

    @staticmethod
    def videos_from_playlist(link):  # download videos from playlist
        driver = YoutubeScrapping.create_driver()
        driver.get(link)
        time.sleep(2)

        pyautogui.FAILSAFE = False
        pyautogui.moveTo((pyautogui.size().width) / 2, (pyautogui.size().height) / 2)
        for i in range(5):
            pyautogui.scroll(-999999999)
            time.sleep(2)

        content = driver.page_source
        driver.quit()
        Parsers = ['lxml', 'xml', 'html.parser']
        titlesList = []
        Urllist = []

        for i in range(3):
            soup = BeautifulSoup(content, Parsers[i])
            titles_Url = soup.findAll('a', id='video-title')

            for title in titles_Url:
                titlesList.append(title.text)
                Urllist.append(title.get('href'))

        titlesList = YoutubeScrapping.unique(titlesList)
        Urllist = YoutubeScrapping.unique(Urllist)

        for title, Url in zip(titlesList, Urllist):
            print(title, '--->', 'https://www.youtube.com' + Url)
        print(len(Urllist))

        TotalStart = time.time()
        for i in range(len(Urllist)):
            start = time.time()
            yt = YouTube(Urllist[i])
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            out_file = video.download(output_path="./Videos")
            base, ext = os.path.splitext(out_file)
            random_number = random.randint(0, 999999999)
            new_file = base + str(random_number) + '.mp4'
            os.rename(out_file, new_file)
            stop = time.time()
            Time = stop - start
            print("Done In ->> ", Time, "s")
        TotalStop = time.time()

        TotalTime = TotalStop - TotalStart
        return print("Total Time is : ", TotalTime)

    @staticmethod
    def audios_from_playlist(link):  # download audios from playlist
        driver = YoutubeScrapping.create_driver()
        driver.get(link)
        time.sleep(2)

        pyautogui.FAILSAFE = False
        pyautogui.moveTo((pyautogui.size().width) / 2, (pyautogui.size().height) / 2)
        for i in range(5):
            pyautogui.scroll(-999999999)
            time.sleep(2)

        content = driver.page_source
        driver.quit()
        Parsers = ['lxml', 'xml', 'html.parser']
        titlesList = []
        Urllist = []

        for i in range(3):
            soup = BeautifulSoup(content, Parsers[i])
            titles_Url = soup.findAll('a', id='video-title')

            for title in titles_Url:
                titlesList.append(title.text)
                Urllist.append(title.get('href'))

        titlesList = YoutubeScrapping.unique(titlesList)
        Urllist = YoutubeScrapping.unique(Urllist)

        for title, Url in zip(titlesList, Urllist):
            print(title, '--->', 'https://www.youtube.com' + Url)
        print(len(Urllist))

        TotalStart = time.time()
        for i in range(len(Urllist)):
            start = time.time()
            yt = YouTube(Urllist[i])
            audio = yt.streams.filter(only_audio=True).first()
            out_file = audio.download(output_path="./Audios")
            base, ext = os.path.splitext(out_file)
            random_number = random.randint(0, 999999999)
            new_file = base + str(random_number) + '.mp3'
            os.rename(out_file, new_file)
            stop = time.time()
            Time = stop - start
            print("Done In ->> ", Time, "s")
        TotalStop = time.time()

        TotalTime = TotalStop - TotalStart
        return print("Total Time is : ", TotalTime)

    @staticmethod
    def video_miner_from_title(title):  # download the top search results (videos) from search query
        url = "https://www.youtube.com/results?search_query="
        CreativeCommon_topViews = "&sp=CAMSAjAB"
        # CreativeCommon_topViews = "&sp=CAM%253D"
        UserInput = title
        url = url + UserInput + CreativeCommon_topViews
        url = url.replace(' ', '+')
        driver = YoutubeScrapping.create_driver()
        driver.get(url)
        time.sleep(2)

        pyautogui.FAILSAFE = False
        pyautogui.moveTo((pyautogui.size().width) / 2, (pyautogui.size().height) / 2)
        for i in range(5):
            pyautogui.scroll(-999999999)
            time.sleep(2)

        content = driver.page_source
        driver.quit()
        Parsers = ['lxml', 'xml', 'html.parser']
        titlesList = []
        Urllist = []

        for i in range(3):
            soup = BeautifulSoup(content, Parsers[i])
            titles_Url = soup.findAll('a', id='video-title')

            for title in titles_Url:
                titlesList.append(title.text)
                Urllist.append(title.get('href'))

        titlesList = YoutubeScrapping.unique(titlesList)
        Urllist = YoutubeScrapping.unique(Urllist)

        for title, Url in zip(titlesList, Urllist):
            print(title, '--->', 'https://www.youtube.com' + Url)
        print(len(Urllist))

        TotalStart = time.time()
        for i in range(len(Urllist)):
            start = time.time()
            yt = YouTube(Urllist[i])
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            out_file = video.download(output_path="./Videos")
            base, ext = os.path.splitext(out_file)
            random_number = random.randint(0, 999999999)
            new_file = base + str(random_number) + '.mp4'
            os.rename(out_file, new_file)
            stop = time.time()
            Time = stop - start
            print("Done In ->> ", Time, "s")
        TotalStop = time.time()

        TotalTime = TotalStop - TotalStart
        return print("Total Time is : ", TotalTime)

    @staticmethod
    def audio_miner_from_title(title):  # download the top search results (audios) from search query
        url = "https://www.youtube.com/results?search_query="
        CreativeCommon_topViews = "&sp=CAMSAjAB"
        UserInput = title
        url = url + UserInput + CreativeCommon_topViews
        url = url.replace(' ', '+')
        driver = YoutubeScrapping.create_driver()
        driver.get(url)
        time.sleep(2)

        pyautogui.FAILSAFE = False
        pyautogui.moveTo((pyautogui.size().width) / 2, (pyautogui.size().height) / 2)
        for i in range(5):
            pyautogui.scroll(-999999999)
            time.sleep(2)

        content = driver.page_source
        driver.quit()
        Parsers = ['lxml', 'xml', 'html.parser']
        titlesList = []
        Urllist = []

        for i in range(3):
            soup = BeautifulSoup(content, Parsers[i])
            titles_Url = soup.findAll('a', id='video-title')

            for title in titles_Url:
                titlesList.append(title.text)
                Urllist.append(title.get('href'))

        titlesList = YoutubeScrapping.unique(titlesList)
        Urllist = YoutubeScrapping.unique(Urllist)

        for title, Url in zip(titlesList, Urllist):
            print(title, '--->', 'https://www.youtube.com' + Url)
        print(len(Urllist))

        TotalStart = time.time()
        for i in range(len(Urllist)):
            start = time.time()
            yt = YouTube(Urllist[i])
            audio = yt.streams.filter(only_audio=True).first()
            out_file = audio.download(output_path="./Audios")
            base, ext = os.path.splitext(out_file)
            random_number = random.randint(0, 999999999)
            new_file = base + str(random_number) + '.mp3'
            os.rename(out_file, new_file)
            stop = time.time()
            Time = stop - start
            print("Done In ->> ", Time, "s")
        TotalStop = time.time()

        TotalTime = TotalStop - TotalStart
        return print("Total Time is : ", TotalTime)

    @staticmethod
    def video_miner_from_txt(file_name):  # download the top search results (videos) from .txt file
        topics = []
        file = open(file_name, "r", encoding="utf-8")

        for line in file:
            topics.append(line)

        Urls = []
        for topic in topics:
            url = "https://www.youtube.com/results?search_query="
            CreativeCommon_topViews = "&sp=CAMSAjAB"
            Urls.append(url + topic + CreativeCommon_topViews)

        for i in range(len(Urls)):
            Urls[i] = Urls[i].replace(' ', '+')
            driver = YoutubeScrapping.create_driver()
            driver.get(Urls[i])
            time.sleep(2)

            pyautogui.FAILSAFE = False
            pyautogui.moveTo((pyautogui.size().width) / 2, (pyautogui.size().height) / 2)
            for i in range(5):
                pyautogui.scroll(-999999999)
                time.sleep(2)

            content = driver.page_source
            driver.quit()
            Parsers = ['lxml', 'xml', 'html.parser']
            titlesList = []
            Urllist = []

            for i in range(3):
                soup = BeautifulSoup(content, Parsers[i])
                titles_Url = soup.findAll('a', id='video-title')

                for title in titles_Url:
                    titlesList.append(title.text)
                    Urllist.append(title.get('href'))

            titlesList = YoutubeScrapping.unique(titlesList)
            Urllist = YoutubeScrapping.unique(Urllist)

            for title, Url in zip(titlesList, Urllist):
                print(title, '--->', 'https://www.youtube.com' + Url)
            print(len(Urllist))

            TotalStart = time.time()
            for i in range(len(Urllist)):  # len(Urllist)
                start = time.time()
                yt = YouTube(Urllist[i])
                video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                out_file = video.download(output_path="./Videos")
                base, ext = os.path.splitext(out_file)
                random_number = random.randint(0, 999999999)
                new_file = base + str(random_number) + '.mp4'
                os.rename(out_file, new_file)
                stop = time.time()
                Time = stop - start
                print("Done In ->> ", Time, "s")
            TotalStop = time.time()

            TotalTime = TotalStop - TotalStart
            print("Total Time is : ", TotalTime)

    @staticmethod
    def audio_miner_from_txt(file_name):  # download the top search results (audios) from .txt file
        topics = []
        file = open(file_name, "r", encoding="utf-8")

        for line in file:
            topics.append(line)

        Urls = []
        for topic in topics:
            url = "https://www.youtube.com/results?search_query="
            CreativeCommon_topViews = "&sp=CAMSAjAB"
            Urls.append(url + topic)

        for i in range(len(Urls)):
            Urls[i] = Urls[i].replace(' ', '+')
            driver = YoutubeScrapping.create_driver()
            driver.get(Urls[i])
            time.sleep(2)

            pyautogui.FAILSAFE = False
            pyautogui.moveTo((pyautogui.size().width) / 2, (pyautogui.size().height) / 2)
            for i in range(5):
                pyautogui.scroll(-999999999)
                time.sleep(2)

            content = driver.page_source
            driver.quit()
            Parsers = ['lxml', 'xml', 'html.parser']
            titlesList = []
            Urllist = []

            for i in range(3):
                soup = BeautifulSoup(content, Parsers[i])
                titles_Url = soup.findAll('a', id='video-title')

                for title in titles_Url:
                    titlesList.append(title.text)
                    Urllist.append(title.get('href'))

            titlesList = YoutubeScrapping.unique(titlesList)
            Urllist = YoutubeScrapping.unique(Urllist)

            for title, Url in zip(titlesList, Urllist):
                print(title, '--->', 'https://www.youtube.com' + Url)
            print(len(Urllist))

            TotalStart = time.time()
            for i in range(len(Urllist)):  # len(Urllist)
                start = time.time()
                yt = YouTube(Urllist[i])
                audio = yt.streams.filter(only_audio=True).first()
                out_file = audio.download(output_path="./Audios")
                base, ext = os.path.splitext(out_file)
                random_number = random.randint(0, 999999999)
                new_file = base + str(random_number) + '.mp3'
                os.rename(out_file, new_file)
                stop = time.time()
                Time = stop - start
                print("Done In ->> ", Time, "s")
            TotalStop = time.time()

            TotalTime = TotalStop - TotalStart
            print("Total Time is : ", TotalTime)

    @staticmethod
    def upload_video_to_youtube():
        """
        Upload Automatic to YouTube it will work only if your screen size is (width=1600, height=900)
        use print(pyautogui.size()) to know your screen size.
        you can change all the pyautogui.moveTo(, ) parameters to make this function work properly on your screen.
        just use print(pyautogui.position())  to know the cursor position and change it by using pyautogui.moveTo(, ).
        """
        video_titles = []
        random.shuffle(video_titles)
        pyautogui.FAILSAFE = False
        driver = YoutubeScrapping.create_driver()
        driver.get("") # channel link on select videos panel
        time.sleep(5)
        print(pyautogui.position())
        pyautogui.moveTo(898, 424) # email field
        time.sleep(5)
        pyautogui.click(button='left') # email field
        time.sleep(5)
        pyautogui.write('', interval=0.1) # email field
        time.sleep(5)
        print(pyautogui.position())
        pyautogui.moveTo(947, 621) # next button
        time.sleep(5)
        pyautogui.click(button='left') # next button
        time.sleep(10)  ##############################
        print(pyautogui.position())
        pyautogui.moveTo(913, 449) # password field
        time.sleep(5)
        pyautogui.click(button='left')  # password field
        time.sleep(5)
        pyautogui.write('', interval=0.1)   # password field
        time.sleep(5)
        print(pyautogui.position())
        pyautogui.moveTo(936, 574) # next button
        time.sleep(5)
        pyautogui.click(button='left') # next button
        time.sleep(5)
        pyautogui.moveTo(819, 813) # skip to youtube studio
        time.sleep(5)
        pyautogui.click(button='left') # skip to youtube studio
        time.sleep(5)
        print(pyautogui.position())
        pyautogui.moveTo(796, 576) # select files
        time.sleep(5)
        pyautogui.click(button='left') # select files
        time.sleep(5)
        print(pyautogui.position())
        pyautogui.moveTo(520, 416) # directory field
        time.sleep(5)
        pyautogui.click(button='left') # directory field
        time.sleep(5)
        pyautogui.write("", interval=0.1)
        time.sleep(5)
        pyautogui.moveTo(646, 446)  # open field
        time.sleep(5)
        pyautogui.click(button='left')  # open field
        time.sleep(5)
        pyautogui.moveTo(228, 183)  # move to the first video
        time.sleep(5)
        pyautogui.click(button='left')  # click on the first video
        time.sleep(5)
        pyautogui.moveTo(646, 446)  # open field
        time.sleep(5)
        pyautogui.click(button='left')  # open field
        time.sleep(10)
        pyautogui.moveTo(803, 366)  # move to the title field
        time.sleep(5)
        pyautogui.write(video_titles[random.randint(0,4)], interval=0.1)
        time.sleep(5)
        pyautogui.moveTo(1238, 775) # next
        time.sleep(5)
        pyautogui.click(button='left')
        time.sleep(5)
        pyautogui.click(button='left')
        time.sleep(5)
        pyautogui.click(button='left')
        time.sleep(5)
        pyautogui.click(button='left')
        time.sleep(6)
        driver.quit()
        print(pyautogui.position())

    @staticmethod
    def trim_random_1min_from_audios(audio_name):
        """
        trim 1 minute from audio files (length > 1 minute) that in Audios folder
        and for the audios that their length is < 1 minute trim those audios into 10 second clips.
        """
        AudioSegment.ffmpeg = "ffmpeg.exe"
        AudioSegment.converter = "ffmpeg.exe"
        AudioSegment.ffprobe = "ffprobe.exe"

        directory = "1_min_audios"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)

        directory10 = "10_sec_audios"
        path10 = os.path.join(parent_dir, directory10)

        try:
            os.mkdir(path)
            os.mkdir(path10)
        except:
            print("Folder already exist")

        sound = AudioSegment.from_file(os.getcwd() + "\\Audios\\" + audio_name)

        length = sound.duration_seconds
        print("duration sec: " + str(length))
        print("duration min: " + str(int(length / 60)) + ':' + str(int(length % 60)))

        checker = False

        if int(length / 60) > 1:
            while checker == False:
                try:
                    StartMin = random.randint(0, int(length / 60) - 1)
                    StartSec = random.randint(0, int(length % 60) - 1)
                    EndMin = StartMin + 1
                    EndSec = StartSec
                    StartTime = StartMin * 60 * 1000 + StartSec * 1000
                    EndTime = EndMin * 60 * 1000 + EndSec * 1000
                    extract = sound[StartTime:EndTime]
                    base, ext = os.path.splitext(audio_name)
                    new_audio = ".\\1_min_audios\\" + base + "_1min.mp3"
                    extract.export(new_audio, format="mp3")
                    checker = True
                except:
                    print("the length of the audio is smaller than 1 minute -->> it will be trimmed to 10 second clips")
                    checker = True
        else:
            print("the length of the audio is smaller than 1 minute -->> it will be trimmed to 10 second clips")
            StartMin = 0
            StartSec = 0
            EndMin = 0
            EndSec = 10
            StartTime = StartMin * 60 * 1000 + StartSec * 1000
            EndTime = EndMin * 60 * 1000 + EndSec * 1000
            extract = sound[StartTime:EndTime]
            base, ext = os.path.splitext(audio_name)
            new_audio = ".\\10_sec_audios\\" + base + "_" + str(random.randint(0, 999999999)) + "_10sec.mp3"
            extract.export(new_audio, format="mp3")

            while int(length % 60) > (EndSec + 10):
                StartMin = 0
                StartSec = StartSec + 10
                EndMin = 0
                EndSec = EndSec + 10
                StartTime = StartMin * 60 * 1000 + StartSec * 1000
                EndTime = EndMin * 60 * 1000 + EndSec * 1000
                extract = sound[StartTime:EndTime]
                base, ext = os.path.splitext(audio_name)
                new_audio = ".\\10_sec_audios\\" + base + "_" + str(random.randint(0, 999999999)) + "_10sec.mp3"
                extract.export(new_audio, format="mp3")

    @staticmethod
    def trim_random_10sec_from_audios(audio_name):
        directory = "10_sec_audios"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)

        try:
            os.mkdir(path)
        except:
            print("Folder already exist")

        sound = AudioSegment.from_file(os.getcwd() + "\\Audios\\" + audio_name)

        length = sound.duration_seconds
        print("duration sec: " + str(length))
        print("duration min: " + str(int(length / 60)) + ':' + str(int(length % 60)))

        # StartMin = 0
        # StartSec = 0
        # EndMin = 0
        # EndSec = 10
        # StartTime = StartMin * 60 * 1000 + StartSec * 1000
        # EndTime = EndMin * 60 * 1000 + EndSec * 1000
        # extract = sound[StartTime:EndTime]
        # base, ext = os.path.splitext(audio_name)
        # new_audio = ".\\10_sec_audios\\" + base + "_" + str(random.randint(0, 999999999)) + "_10sec.mp3"
        # extract.export(new_audio, format="mp3")

        for i in range(1):
            StartMin = random.randint(0,(int(length / 60)-1))
            StartSec = 0
            EndMin = StartMin
            EndSec = 10
            StartTime = StartMin * 60 * 1000 + StartSec * 1000
            EndTime = EndMin * 60 * 1000 + EndSec * 1000
            extract = sound[StartTime:EndTime]
            base, ext = os.path.splitext(audio_name)
            new_audio = ".\\10_sec_audios\\" + base + "_" + str(random.randint(0, 999999999)) + "_10sec.mp3"
            extract.export(new_audio, format="mp3")

    @staticmethod
    def trim_random_1min_from_videos(video_name):  # trim a random 1 minute from videos
        directory = "1_min_videos"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)

        try:
            os.mkdir(path)
        except:
            print("Folder already exist")

        video = AudioSegment.from_file(os.getcwd() + "\\Videos\\" + video_name)

        length = video.duration_seconds
        print("duration sec: " + str(length))
        print("duration min: " + str(int(length / 60)) + ':' + str(int(length % 60)))

        checker = False

        if int(length / 60) > 1:
            while checker == False:
                try:
                    StartMin = random.randint(0, int(length / 60) - 1)
                    StartSec = random.randint(0, int(length % 60) - 1)

                    EndMin = StartMin + 1
                    EndSec = StartSec

                    StartMin = StartMin * 60
                    EndMin = EndMin * 60

                    base, ext = os.path.splitext(video_name)
                    new_video = ".\\1_min_videos\\" + base + "_" + str(random.randint(0, 999999999)) + "_1min.mp4"
                    file_location = os.getcwd() + "\\Videos\\" + video_name
                    print("start second is : ", (StartMin + StartSec), "\nEnd second is : ", (EndMin + EndSec))
                    ffmpeg_extract_subclip(file_location, (StartMin + StartSec), (EndMin + EndSec), targetname=new_video)

                    checker = True
                except:
                    print("the length of the video is smaller than 1 minute")
                    checker = True
        else:
            print("the length of the video is smaller than 1 minute")

    @staticmethod
    def change_video_background_music(video_name, audio_name): # change the background music of the video
        input_video = os.getcwd() + "\\1_min_videos\\" + video_name
        input_audio = os.getcwd() + "\\1_min_audios\\" + audio_name


        directory = "edited_videos"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)

        try:
            os.mkdir(path)
        except:
            print("Folder already exist")

        base, ext = os.path.splitext(video_name)
        new_video = ".\\edited_videos\\" + base + "_" + str(random.randint(0, 999999999)) + "_edited.mp4"
        output_video = new_video

        video_clip = mymovie.VideoFileClip(input_video)
        audio_clip = mymovie.AudioFileClip(input_audio)
        edited_clip = video_clip.set_audio(audio_clip)
        edited_clip.write_videofile(output_video, fps=60)

    @staticmethod
    def get_proxies1():
        link = 'https://www.proxy-list.download/HTTP'
        edge_path = '.\msedgedriver.exe'
        browser_options = Options()
        browser_options.add_argument('--no-sanbox')
        browser_options.add_argument('headless')
        browser_options.add_argument('disable-notificatioon')
        browser_options.add_argument('--disable-infobars')
        browser_options.add_argument("--start-maximized")
        proxies = []
        td = []

        driver = webdriver.Edge(executable_path=edge_path, options=browser_options)
        driver.get(link)
        driver.execute_script("window.scrollTo(0, 4000)")
        time.sleep(2)

        for i in range(10):
            content = driver.page_source
            Parsers = ['lxml', 'xml', 'html.parser']

            soup = BeautifulSoup(content, Parsers[0])
            page = soup.find_all("tr")
            for item in page:
                td.append(item.find_all("td"))

            td.pop(0)
            for ele in td:
                proxies.append(ele[0].text + ":" + ele[1].text)

            element = driver.find_element(By.XPATH, ".//li[2]//button")
            # print(element.get_attribute("onclick"))
            try:
                element.click()
                print("Done")
            except:
                print("can't press the next button")
                pass

            td = []

        for i in range(len(proxies)):
            proxies[i] = proxies[i].replace(" ", "")
            proxies[i] = proxies[i].replace("\n", "")

        driver.quit()

        proxies = YoutubeScrapping.unique(proxies)

        return proxies

    @staticmethod
    def get_proxies2():
        link = 'https://free-proxy-list.net/'
        edge_path = '.\msedgedriver.exe'
        browser_options = Options()
        browser_options.add_argument('--no-sanbox')
        browser_options.add_argument('headless')
        browser_options.add_argument('disable-notificatioon')
        browser_options.add_argument('--disable-infobars')
        browser_options.add_argument("--start-maximized")
        proxies = []
        td = []

        driver = webdriver.Edge(executable_path=edge_path, options=browser_options)
        driver.get(link)
        time.sleep(2)

        content = driver.page_source
        Parsers = ['lxml', 'xml', 'html.parser']

        soup = BeautifulSoup(content, Parsers[0])
        page = soup.find_all("tr")

        for item in page:
            td.append(item.find_all("td"))

        td.pop(0)

        for i in range(len(td)):
            try:
                proxies.append(td[i][0].text + ":" + td[i][1].text)
                print("Done")
            except:
                print("Error in this proxy")
                break

        for i in range(len(proxies)):
            proxies[i] = proxies[i].replace(" ", "")
            proxies[i] = proxies[i].replace("\n", "")

        driver.quit()

        proxies = YoutubeScrapping.unique(proxies)

        return proxies

    @staticmethod
    def test_the_working_proxies(proxy):
        headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"}
        try:
            r = requests.get("https://snaptik.app/en", headers = headers, proxies={'http': proxy, 'https': proxy}, timeout=5)
            print(r.json(), ' | Works')
        except:
            print("this proxy is not working on this link")
            pass

        return proxy

    @staticmethod
    def tiktok_videos_links_miner_from_title(title):
        link = 'https://www.tiktok.com'
        pathEdge = '.\msedgedriver.exe'

        browser_options = Options()
        browser_options.add_argument('--no-sanbox')
        browser_options.add_argument('disable-notificatioon')
        browser_options.add_argument('--disable-infobars')
        browser_options.add_argument("--start-maximized")

        driver = webdriver.Edge(executable_path=pathEdge, options=browser_options)
        driver.get(link)

        tagarSearch = "#" + title
        element = driver.find_element(By.XPATH, "/html/body/div[2]/div[1]/div/div[1]/div/form/input")
        element.send_keys(tagarSearch)
        element.send_keys(Keys.ENTER)
        time.sleep(5)

        # Top menu
        ii = 0
        while ii < 1:
            try:
                driver.find_element(By.XPATH, "/html/body/div[2]/div[2]/div[2]/div[1]/div/div[1]/div[1]/div[1]").click()
                ii = 1
            except:
                ii = 0
                time.sleep(5)

        # Load More
        i = 0
        while i < 10:
            try:
                NextStory = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div[2]/div[2]/div[2]/div[2]/button")))
                NextStory.click()
                time.sleep(2)
            except:
                i = 10

        html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        soup = BeautifulSoup(html, 'html.parser')

        video_links = []

        # link
        for link in soup.find_all('div', class_='tiktok-yz6ijl-DivWrapper e1cg0wnj1'):
            video_links.append(link.a['href'])

        driver.close()

        return video_links

    @staticmethod
    def tiktok_videos_links_converter(list_of_links):
        directory = "tiktok_videos"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)

        try:
            os.mkdir(path)
        except:
            print("Folder already exist")

        option = webdriver.ChromeOptions()
        # option.binary_location = "C:/Program Files/Google/Chrome Beta/Application/chrome.exe"
        # option.add_argument("window-size=1400,600")
        option.add_experimental_option("useAutomationExtension", False)
        option.add_experimental_option("excludeSwitches", ["enable-automation"])

        # avoiding detection
        option.add_argument('--disable-blink-features=AutomationControlled')
        option.add_argument("disable-infobars");
        option.add_argument("--start-maximized");
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'

        option.add_argument("--profile-directory=Default")
        option.add_argument("--user-data-dir=C:\\Users\\Hosam\\AppData\\Local\\Google\\Chrome\\User Data") # user data location
        option.add_argument('headless')

        option.add_argument('user-agent={0}'.format(user_agent))
        ua = UserAgent()
        header = ua.chrome
        option.add_argument(f'user-agent={header}')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)

        link = 'https://snaptik.app/en'

        Url_list1 = []
        Url_list2 = []

        driver.get(link)

        for i in range(10):
            try:
                link_box = driver.find_element("id", "url")
                link_box.send_keys(list_of_links[i])
                time.sleep(2)

                butoon = driver.find_element("id", "submiturl")
                butoon.click()
                time.sleep(7)
                content = driver.page_source

                Parsers = ['lxml', 'xml', 'html.parser']

                for i in range(3):
                    soup = BeautifulSoup(content, Parsers[i])
                    page_source1 = soup.findAll('a', title="Download Server 01")
                    page_source2 = soup.findAll('a', title="Download Server 02")

                    for vid_link in page_source1:
                        Url_list1.append(vid_link.get('href'))

                    for vid_link in page_source2:
                        Url_list2.append(vid_link.get('href'))

                element = driver.find_element(By.XPATH, "//a[@title='Download other video']")
                element.click()
                time.sleep(3)
                print("Done")
            except:
                print("failed")
                pass

        Url_list1 = YoutubeScrapping.unique(Url_list1)
        Url_list2 = YoutubeScrapping.unique(Url_list2)

        driver.quit()

        return Url_list1, Url_list2

    @staticmethod
    def download_url(args):
        t0 = time.time()
        url, fn = args[0], args[1]
        try:
            r = requests.get(url)
            with open(fn, 'wb') as f:
                f.write(r.content)
            return (url, time.time() - t0)
        except Exception as e:
            print('Exception in download_url():', e)

    @staticmethod
    def download_parallel(args):
        cpus = cpu_count()
        results = ThreadPool(cpus - 1).imap_unordered(YoutubeScrapping.download_url, args)
        for result in results:
            print('url:', result[0], 'time (s):', result[1])

##################################################################################################### Main
obj = YoutubeScrapping()
obj.mkdir()

# for i in []:
#     try:
#         obj.audio_from_link(i)
#     except:
#         pass


ar_audios = os.listdir("Audios")
for j in range(30):
    for i in range(len(ar_audios)):
        try:
            obj.trim_random_10sec_from_audios(ar_audios[i])
        except:
            print("error")


############################################### automatic uploading to youtube
# while True:  # automatic upload a video to YouTube every 1 - 2 hours
#     obj.rename_videos()
#     obj.upload_video_to_youtube()
#     obj.remove_vid1()
#     obj.rename_videos()
#     time.sleep(random.randint(3600, 7200))

################################################ automatic downloading from tiktok
# ls_links = obj.tiktok_videos_links_miner_from_title("woodcraft")
# print(ls_links[0])
# server1, server2 = obj.tiktok_videos_links_converter(ls_links)
#
# print(len(server1))
# print(len(server2))
#
# directory = "tiktok_videos"
# parent_dir = os.getcwd()
# path = os.path.join(parent_dir, directory)
#
# video_names = []
# for i in range(len(server1)):
#     video_names.append(path + "\\vid_" + str(random.randint(0, 999999999)) + "_.mp4")
#
# inputs = zip(server1, video_names)
#
# obj.download_parallel(inputs)

################################################ test proxies
# proxies1, proxies2  = obj.get_proxies1(), obj.get_proxies2()
#
# proxies1 = proxies1 + proxies2
# proxies1 = obj.unique(proxies1)
#
# print(len(proxies1))
# print(proxies1)
#
# working_proxies = []
#
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     executor.map(obj.test_the_working_proxies, proxies1)
#
# print(len(working_proxies))
# print(working_proxies)

# from moviepy.editor import VideoFileClip, concatenate_videoclips
# clip1 = VideoFileClip("vid_41522857_.mp4")
# clip2 = VideoFileClip("vid_43123592_.mp4")
# clip3 = VideoFileClip("vid_83093271_.mp4")
# final_clip = concatenate_videoclips([clip1,clip2,clip3])
# final_clip.write_videofile("my_concatenation.mp4")