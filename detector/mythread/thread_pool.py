# !/usr/bin/env python
# -*- coding:utf-8 -*-

import Queue
import job
import time;


class JobTaskManager(object):
    def __init__(self, job_num=5, thread_num=2):
        self.work_queue = Queue.Queue()
        self.threads = []
        self.job_num = job_num;
        self.__init_thread_pool(thread_num)

    def __init_thread_pool(self, thread_num):
        for i in range(thread_num):
            self.threads.append(job.Job(self.work_queue))

    def addJobTask(self, func, *args):
        self.work_queue.put((func, list(args)))
        # print 'aaaaaaaaa', self.work_queue.qsize()
        self._checkThread()
        while self.work_queue.qsize() >= self.job_num:
            time.sleep(0.001);

    def getthreadStatus(self):
        for index, item in enumerate(self.threads):
            if not item.getEventStatus():
                return False;
        return True;  # 所有程序正常运行完

    def _checkThread(self):
        """
        检测线程是否还活着
        :return:
        """
        status = False;
        for index, item in enumerate(self.threads):
            if item.isAlive() and item.getEventStatus():
                if not item.isAlive():
                    print item, 'all thread are dead !'
                    continue;
                self.threads[index].restart()
                status = True;
        return status;

    def _join_all(self):
        for item in self.threads:
            item.unlock();
            if item.isAlive():
                item.join()
