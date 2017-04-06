# -*-coding:utf-8-*-
import threading as _th
import time


class Job(_th.Thread):
    def __init__(self, work_queue=None):
        _th.Thread.__init__(self)
        self.ev = _th._Event()
        self._event_status = False;  # 是否挂起
        self.var_lock = True;
        self.work_queue = work_queue
        self.mutex = _th.Lock()  # 互斥锁
        self.start();

    def getEventStatus(self):
        return self._event_status;

    def restart(self):
        self._event_status = False;
        self.ev.set();

    def unlock(self):
        self.var_lock = False;
        self.restart()

    def getlockstatus(self):
        return self.var_lock;

    def run(self):
        """
          直到程序执行完.
        :return:
        """

        while self.getlockstatus():
            if self.work_queue is not None and self.work_queue.qsize() > 0:
                function, args = self.work_queue.get()  # 非阻塞模式.block=False
                function(args)
                self.work_queue.task_done()  # 通知系统任务完成
            else:
                if self.ev.is_set():
                    self.ev.clear();
                self._event_status = True;
                self.ev.wait();
