import logging  
import logging.handlers  
     
class Logger():  
    def __init__(self, tag="", filename="python.log", loglevel=1, toConsole=True, toFile=True):  
        ''''' 
           既要把日志输出到控制台， 还要写入日志文件 
        '''  
        # 创建一个logger  
        # 返回一个logger实例，如果没有指定name，返回root logger。  
        # 只要name相同，返回的logger实例都是同一个而且只有一个，即name和logger实例是一一对应的。  
        # 这意味着，无需把logger实例在各个模块中传递。只要知道name，就能得到同一个logger实例。  
        # tag就是Logger的名字，相当于Android中的TAG  
        self.logger = logging.getLogger(tag)  
          
        # 防止重复记录日志的问题  
        if not self.logger.handlers:  
              
            # 定义handler的输出格式：  
            #   [时间][代码文件名,函数名:行号][logger名(相当于Android中的TAG)]: 输出日志信息  
            format_dict = {  
               1 : logging.Formatter('[%(asctime)s][%(filename)s,%(funcName)s:%(lineno)s][%(name)s]: %(message)s'),  
               2 : logging.Formatter('[%(asctime)s][%(filename)s,%(funcName)s:%(lineno)s][%(name)s]: %(message)s'),  
               3 : logging.Formatter('[%(asctime)s][%(filename)s,%(funcName)s:%(lineno)s][%(name)s]: %(message)s'),  
               4 : logging.Formatter('[%(asctime)s][%(filename)s,%(funcName)s:%(lineno)s][%(name)s]: %(message)s'),  
               5 : logging.Formatter('[%(asctime)s][%(filename)s,%(funcName)s:%(lineno)s][%(name)s]: %(message)s')  
            }  
            formatter = format_dict[int(loglevel)]  
              
            # 指定最低的日志级别，低于指定级别的将被忽略。  
            # 级别高低顺序：NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL  
            # 如果把looger的级别设置为INFO， 那么小于INFO级别的日志都不输出， 大于等于INFO级别的日志都输出　  
            self.logger.setLevel(logging.DEBUG)  
  
            # 创建一个handler，用于写入日志文件  
            # 需要创建/写文件的权限  
            try:  
                if toFile:  
                    fh = logging.FileHandler(filename)  
                    fh.setLevel(logging.DEBUG)  
                    fh.setFormatter(formatter)  
                    self.logger.addHandler(fh)  
            except Exception as e:  
                print(e)  
              
            # 再创建一个handler，用于输出到控制台  
            if toConsole:  
                ch = logging.StreamHandler()  
                ch.setLevel(logging.DEBUG)  
                ch.setFormatter(formatter)  
                self.logger.addHandler(ch)  
     
    def getlogger(self):  
        return self.logger  
  
          
def LOG(file, s):  
    Logger(filename = file).getlogger().debug(s)  
  
      


