
import sys
import time

from wetb.utils.ui import OutputUI, InputUI, StatusUI


class TextOutputUI(OutputUI):
    last_text = ""
    
    def __init__(self, parent=None):
        OutputUI.__init__(self, parent=parent)
        self.errors = sys.stdout.errors
        self.encoding = sys.stdout.encoding
        sys.stdout = self
        sys.stderr = self
        
        
        
    def show_message(self, msg, title="Information"):
        #self.last_text = msg
        if title != "":
            print ("\n\n%s\n%s\n%s\n" % (title, "-"*len(title), msg))
        else:
            print (msg)

    def show_warning(self, msg, title="Warning"):
        #self.last_text = msg
        print ("%s\n%s\n%s" % (title, "-"*len(title), msg))

    def show_text(self, text, end="\n", flush=False):
        #self.last_text = text
        #print (text, end=end)
        self.write(text+end)
        if flush:
            sys.stdout.flush()
            
    def flush(self):
        sys.__stdout__.flush()
            
    def write(self, txt):
        if txt.strip()!="":
            self.last_text = txt
        sys.__stdout__.write(txt)
        

class TextInputUI(InputUI):
    def get_confirmation(self, title, msg):
        raise NotImplementedError

    def get_string(self, title, msg):
        raise NotImplementedError

    def get_open_filename(self, title="Open", filetype_filter="*.*", file_dir=None, selected_filter=None):
        raise NotImplementedError
    def get_save_filename(self, title, filetype_filter, file_dir=None, selected_filter=None):
        raise NotImplementedError


    def get_open_filenames(self, title, filetype_filter, file_dir=None):
        raise NotImplementedError

    def get_foldername(self, title='Select directory', file_dir=None):
        raise NotImplementedError



class TextStatusUI(StatusUI, TextOutputUI):
    

    def progress_iterator(self, sequence, text="Working... Please wait", allow_cancel=True, always_refresh=False):
        global pct
        it = iter(list(sequence))
        if it.__length_hint__() > 0:
            def init():
                global pct
                self.show_text("\n\n|0" + " " * 46 + "50%" + " " * 46 + "100%|")
                pct = 0
                self.show_text("|", end="")

            self.show_text(text, flush=True)
            #sys.__stdout__.flush()
            N = it.__length_hint__()
            init()
            for n, v in enumerate(it):
                #if n % 100 == 99:
                #    self.show_text("")
                if (self.last_text != "." and n > 0) or (always_refresh and ((n + 1) / N * 100 > pct)):
                    init()
                while ((n + 1) / N * 100 > pct):
                    self.show_text(".", end="", flush=True)
                    pct += 1
                yield(v)
            self.show_text("|")


    def exec_long_task(self, text, allow_cancel, task, *args, **kwargs):
        print (text)
        return task(*args, **kwargs)

    def start_wait(self):
        #print ("Working please wait")
        pass

    def end_wait(self):
        #print ("finish")
        pass


    def progress_callback(self, text="Working... Please wait", always_refresh=False):
        class ProgressCallBack():
            
            def __init__(self, ui, text, always_refresh=False):
                self.ui = ui
                self.text = text
                self.always_refresh = always_refresh
                self.pct = None
            
            def init(self):
                self.ui.show_text("\n\n" + self.text, flush=True)
                self.ui.show_text("|0" + " " * 46 + "50%" + " " * 46 + "100%|")
                self.pct = 0
                self.ui.show_text("|", end="")
                
            def __call__(self, n, N):
                if (self.pct is None) or (self.ui.last_text != "." and self.pct > 0) or (self.always_refresh and ((n + 1) / N * 100 > self.pct)):
                    self.init()
                while ((n + 1) / N * 100 > (self.pct+1)):
                    self.ui.show_text(".", end="", flush=True)
                    self.pct += 1
                if self.pct==100:
                    self.ui.show_text("|")
        return ProgressCallBack(self, text, always_refresh)
                
class TextUI(TextInputUI, TextStatusUI):
    pass


if __name__ == "__main__":
    def task(callback):
        for i in range(100):
            callback(i,100)
            time.sleep(0.05)
            
    task (TextStatusUI().progress_callback())
        