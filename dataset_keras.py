#-*- encoding: utf-8 -*-

import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

csv_path = 'Data/train.csv'
df_label = pd.read_csv(csv_path, index_col=0)
df_label['label_len'] = df_label['label'].apply(len)
MAX_LABEL_LEN = df_label.label_len.max()
df_label['LEN'] = MAX_LABEL_LEN

import string
import random
from captcha.image import ImageCaptcha

symbols = string.digits + '+-*=()'
nClass = len(symbols)

class dataset(object):
    
    def __init__(self, train=True, dir='Data/train_dir/train', size=(64,300), shuffle=False):
        self.train = train
        self.shuffle = shuffle
        self.dir = dir
        self.target_size = size
        self.batch_index = 0
        self.files_origin = os.listdir(self.dir)
        self.seed = 112233
        self.files = self.gen_files()
        self.length = len(self.files)

        self.captcha_gen = ImageCaptcha(width=self.target_size[1], height=self.target_size[0], 
                            font_sizes=range(43, 50), 
                         fonts=['fonts/%s'%x for x in os.listdir('fonts') if '.ttf' in x])


    def gen_files(self, train=True):
        train_files, val_files = train_test_split(self.files_origin, test_size=0.3
                                                  ,random_state=self.seed)
        if self.train:
            return train_files
        return val_files
    
    def get_batch(self, batch_size=50, color_mode='grayscale'):
    
        while True:
            if (self.batch_index+1)*batch_size >= self.length: 
                files = self.files[self.batch_index*batch_size:]               
                self.batch_index = 0
                random.seed(self.seed)
                self.seed = random.randint(0,100)
                self.files = self.gen_files()
            else:
                files = self.files[self.batch_index*batch_size:(self.batch_index+1)*batch_size]
                self.batch_index += 1   
                
            batch_size = len(files)
            
            labels = -1* np.ones((batch_size, 11),dtype=np.int32)
            label_lengths = np.zeros(batch_size)
            imgs = np.zeros((batch_size, 64,300,1),dtype=np.uint8)
            
            for i,file in enumerate(files):
                label = df_label.loc['train/'+file, 'label']
                labels[i,:len(label)] = [symbols.find(c) for c in label]
                label_lengths[i] = len(label)
                img = cv2.imread(os.path.join(self.dir,file),cv2.IMREAD_GRAYSCALE)
                imgs[i] = img[:,:,np.newaxis]
            
            input_lengths = np.array([17]*batch_size)
            yield [imgs, labels, input_lengths, label_lengths], np.ones(batch_size)

    def gen_data(self,batch_size=50):
        
        width, height = 300, 64
        imgs = np.zeros([batch_size, height, width, 1], dtype=np.uint8)
        labels = -1*np.ones([batch_size, 11], dtype=np.int32)
        label_lens = np.zeros(batch_size)
        while True:
            for i in range(batch_size):
                label = self.gen_captcha(3)
                img = np.array(self.captcha_gen.generate_image(label))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgs[i] = img[:,:,np.newaxis]
                labels[i,:len(label)] = [symbols.find(x) for x in label]
                label_lens[i] = len(label)
            yield [imgs, labels, np.ones(batch_size)*int(17), label_lens], np.ones(batch_size)

    def gen_captcha(self, cap_len, bracelet=True,equation=True):
    
        ops = ["+","-","*"]
        bracelet = random.choice([True,False])
        if bracelet:
            braces = [1,1]
            braces[0] = random.choice(range(cap_len-1))
            braces[1] = random.choice(range(braces[0]+1,cap_len))
        else:
            braces = [-1,-1]
        
        def get_factor(num):
            return [str(a) for a in range(1,num+1) if num%a==0]

        def last_index(string,obj,begin=0):
            index = string.find(obj,begin)
            if index == -1:
                return begin-1
            return last_index(string,obj,index+1)
        
        def get_num(tmp):
            if int(eval(tmp)) == 0:
                    num = np.random.choice([str(a) for a in range(1,10)])
            else:
                num = np.random.choice(get_factor(int(eval(tmp))))
            return num

        
        char = []
        num = np.random.choice([str(a) for a in range(10)])
        if braces[0] == 0:
            char.append('(')
        char.append(num)
        
        for i in range(cap_len-1):
            if i==braces[1]:
                char.append(')')
            tmp = ''.join(char) 
            op = np.random.choice(ops)
            char.append(op)

            if (op=='/'):
                if ('(' in tmp) and (')' not in tmp):
                    tmp = tmp[tmp.find('(')+1:] 
                elif ')' in tmp[:-1]:
                    tmp = tmp[tmp.find(')')+1:]
                elif ')' == tmp[-1]:
                    tmp = tmp[tmp.find('(')+1:-1]
                idx = max(last_index(tmp,'+'),last_index(tmp,'-'))
                if idx != -1:
                    tmp = tmp[idx+1:]
                    num = get_num(tmp)
                else:
                    num = get_num(tmp)
            else:
                num = np.random.choice([str(a) for a in range(10)])
                
            if i+1==braces[0]:
                char.append('(')
            char.append(num)

        if braces[1]==cap_len-1:
            char.append(')')
        chars = ''.join(char)
        if chars[chars.find('(')-1] == '/':
            tmp = chars[chars.find('(')+1:chars.find(')')]
            chars1 = chars[:chars.find('(')+1]
            chars2 = chars[chars.find(')'):]
            cap_len = len(tmp)//2+1
            while int(eval(tmp))==0:
                tmp = gen_captcha(cap_len, bracelet=False, equation=False)
            chars = chars1+tmp+chars2
        if equation:
            return chars+'='+str(int(eval(chars)))
        return chars