#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import threading
import psutil
import math

class Progress():
    def __init__(self,width):
        self.width = width
        self.max = 0

    def setMax(self, maximum):
        self.max = maximum

    def start(self):
        sys.stdout.write('\r')
        sys.stdout.write("[%2s/%2s]" % (0, self.max))
        sys.stdout.flush()

    def end(self):
        sys.stdout.write('\n')
        sys.stdout.flush()

    def print(self, progress):
        progress
        if(self.max > 0):
            procentage = progress / self.max
        else:
            procentage = 0

        progress_chars_num = math.floor(self.width * procentage)
        progress_chars = u"\u2588"*(progress_chars_num)
        sys.stdout.write('\r')
        if(progress <= self.max):
            sys.stdout.write("%s [%2s/%2s]" % (progress_chars, progress, self.max))
        else:
            sys.stdout.write("%s [%2s/%2s] Note:some docs are split " % (progress_chars, progress, self.max))

        sys.stdout.flush()

class Timer():
    def __init__(self):
        self.time_start = None
        self.time_stop = None

    def start(self):
        self.time_start = time.perf_counter()

    def stop(self):
        self.time_stop = time.perf_counter()

    def info(self):
        if(self.time_start != None and self.time_stop != None):
            elapsed =  self.time_stop - self.time_start
            self.time_start = None
            self.time_stop = None

            return elapsed
            

        return None

class Tester():
    def __init__(self):
        self.count = 0
        self.running = False
        self.time_start = None
        self.time_stop = None 
        self.cpu_percent_sum = None
        self.memory_percent_sum = None
        self.thread = None

    def _do(self):
        self.cpu_percent_sum = 0
        self.memory_percent_sum = 0
        self.count = 0
        self.running = True
        currentProcess = psutil.Process()

        while self.running:
            cpu_curent =  currentProcess.cpu_percent(interval=1)
            self.cpu_percent_sum += cpu_curent
            memory_curent = currentProcess.memory_percent()
            self.memory_percent_sum += memory_curent
            self.count += 1
    
    def start(self):
        self.time_start = time.perf_counter()
        self.thread =  threading.Thread(target=self._do)
        self.thread.start()
       
    def stop(self):
        self.running = False
        self.thread.join()
        self.time_stop  =  time.perf_counter()
    
    def info(self):
        if(not self.running and self.time_stop != None):
            execution_time = self.time_stop - self.time_start
            if(self.count   > 0):
             
                cpu_percent_avg = self.cpu_percent_sum / self.count  
                memory_percent_avg = self.memory_percent_sum / self.count  
        
            return "Execution time: {} Averge CPU percentage: {}% Average memory utilization percentage: {}%".format(execution_time, cpu_percent_avg, memory_percent_avg )
            
        return None