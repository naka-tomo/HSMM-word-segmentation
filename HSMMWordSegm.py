# encoding: utf8
#from __future__ import unicode_literals
import numpy
import random
import math
import time
import codecs
import os


class HSMMWordSegm():
    MAX_LEN = 8
    AVE_LEN = 2

    def __init__(self, nclass):
        self.num_class = nclass
        self.word_class = {}
        self.segm_sentences = []
        self.trans_prob = numpy.ones( (nclass,nclass) )
        self.trans_prob_bos = numpy.ones( nclass )
        self.trans_prob_eos = numpy.ones( nclass )
        self.word_count = [ {} for i in range(nclass) ]
        self.num_words = []
        self.prob_char = {}
        self.num_vocab = numpy.zeros( self.num_class )

    def load_data(self, filename ):
        self.word_count = {}
        self.num_words = [0]*self.num_class
        self.segm_sentences = []
        self.word_count = [ {} for i in range(self.num_class) ]
        self.sentences = [ line.replace("\n","").replace("\r", "") for line in codecs.open( filename, "r" , "sjis" ).readlines()]
        self.prob_char = {}

        for sentence in self.sentences:
            words = []

            i = 0
            while i<len(sentence):
                # ランダムに切る
                length = random.randint(1,self.MAX_LEN)

                if i+length>=len(sentence):
                    length = len(sentence)-i

                words.append( sentence[i:i+length] )

                i+=length

            self.segm_sentences.append( words )

            # ランダムに割り振る
            for i,w in enumerate(words):
                c = random.randint(0,self.num_class-1)
                self.word_class[id(w)] = c
                self.word_count[c][w] = self.word_count[c].get( w , 0 ) + 1
                self.num_words[c] += 1

        # 遷移確率更新
        self.calc_trans_prob()

        # 文字が発生する確率計算
        sum = 0
        for sentence in self.sentences:
            for ch in sentence:
                self.prob_char[ch] = self.prob_char.get( ch, 0.0 ) + 1.0
                sum += 1

        for ch in self.prob_char.keys():
            self.prob_char[ch] /= sum

        for words in self.segm_sentences:
            for w in words:
                print w,
            print


    def calc_output_prob(self, c , w ):
        conc_param = 10.0
        prior = 1.0
        # 文字発生確率
        for ch in w:
            prior *= self.prob_char[ch]

        # 長さ
        L = len(w)
        prior *= (self.AVE_LEN**L) * math.exp( -self.AVE_LEN ) / math.factorial(L)

        p = ( self.word_count[c].get(w,0) +  conc_param*prior ) / ( self.num_words[c] + conc_param )

        return p

    def forward_filtering(self, sentence ):
        T = len(sentence)
        a = numpy.zeros( (len(sentence), self.MAX_LEN, self.num_class) )                            # 前向き確率

        for t in range(T):
            for k in range(self.MAX_LEN):
                if t-k<0:
                    break

                for c in range(self.num_class):
                    out_prob = self.calc_output_prob( c , sentence[t-k:t+1] )

                    # 遷移確率
                    tt = t-k-1
                    if tt>=0:
                        for kk in range(self.MAX_LEN):
                            for cc in range(self.num_class):
                                a[t,k,c] += a[tt,kk,cc] * self.trans_prob[cc, c]
                        a[t,k,c] *= out_prob
                    else:
                        # 最初の単語
                        a[t,k,c] = out_prob * self.trans_prob_bos[c]

                    # 最後の単語の場合
                    if t==T-1:
                        a[t,k,c] *= self.trans_prob_eos[c]



        return a

    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i

    def backward_sampling(self, a, sentence, use_max_path=False):
        T = a.shape[0]
        t = T-1

        words = []
        classes = []

        c = -1
        while True:

            # 状態cへ遷移する確率
            if c==-1:
                trans = numpy.ones( self.num_class )
            else:
                trans = self.trans_prob[:,c]

            if use_max_path:
                idx = numpy.argmax( (a[t]*trans).reshape( self.MAX_LEN*self.num_class ) )
            else:
                idx = self.sample_idx( (a[t]*trans).reshape( self.MAX_LEN*self.num_class ) )


            k = int(idx/self.num_class)
            c = idx % self.num_class

            w = sentence[t-k:t+1]

            words.insert( 0, w )
            classes.insert( 0, c )

            t = t-k-1

            if t<0:
                break

        return words, classes


    def calc_trans_prob( self ):
        self.trans_prob = numpy.zeros( (self.num_class,self.num_class) ) + 0.1
        self.trans_prob_bos = numpy.zeros( self.num_class ) + 0.1
        self.trans_prob_eos = numpy.zeros( self.num_class ) + 0.1

        # 数え上げる
        for n,words in enumerate(self.segm_sentences):
            try:
                # BOS
                c = self.word_class[ id(words[0]) ]
                self.trans_prob_bos[c] += 1
            except KeyError,e:
                # gibss samplingで除かれているものは無視
                continue

            for i in range(1,len(words)):
                cc = self.word_class[ id(words[i-1]) ]
                c = self.word_class[ id(words[i]) ]

                self.trans_prob[cc,c] += 1.0

            # EOS
            c = self.word_class[ id(words[-1]) ]
            self.trans_prob_eos[c] += 1

        # 正規化
        self.trans_prob = self.trans_prob / self.trans_prob.sum(1).reshape(self.num_class,1)
        self.trans_prob_bos = self.trans_prob_bos / self.trans_prob_bos.sum()
        self.trans_prob_eos = self.trans_prob_eos / self.trans_prob_eos.sum()


    def print_result(self):
        print "-------------------------------"
        for words in self.segm_sentences:
            for w in words:
                print w,"|",
            print

        num_voca = 0
        for c in range(self.num_class):
            num_voca += len(self.word_count[c])

        print num_voca

    def delete_words(self):
        self.num_vocab = numpy.zeros( self.num_class )
        for c in range(self.num_class):
            for w,num in self.word_count[c].items():
                self.num_vocab[c] += 1
                if num==0:
                    self.word_count[c].pop( w )



    def learn(self,use_max_path=False):
        for i in range(len(self.sentences)):
            sentence = self.sentences[i]
            words = self.segm_sentences[i]

            # 学習データから削除
            for w in words:
                c = self.word_class[id(w)]
                self.word_class.pop( id(w) )
                self.word_count[c][w] -= 1
                self.num_words[c] -= 1

            # 遷移確率更新
            self.calc_trans_prob()

            # foward確率計算
            a = self.forward_filtering( sentence )

            # backward sampling
            # words: 分節化された単語
            # classes: 各単語が分類されたクラス
            words, classes = self.backward_sampling( a, sentence, use_max_path )

            # パラメータ更新
            self.segm_sentences[i] = words
            for w,c in zip( words, classes ):
                self.word_class[id(w)] = c
                self.word_count[c][w] = self.word_count[c].get( w , 0 ) + 1
                self.num_words[c] += 1

            # 遷移確率更新
            self.calc_trans_prob()

            # 空の単語を削除
            self.delete_words()


        return

    def save_result(self, dir ):
        if not os.path.exists(dir):
            os.mkdir(dir)

        for c in range(self.num_class):
            path = os.path.join( dir , "word_count_%03d.txt" %c )
            f = codecs.open( path, "w" , "sjis" )
            for w,num in self.word_count[c].items():
                f.write( "%s\t%d\n" % (w,num) )
            f.close()

        path = os.path.join( dir , "result.txt" )
        f = codecs.open( path ,  "w" , "sjis" )
        for words in self.segm_sentences:
            for w in words:
                f.write( w )
                f.write( " | " )

            f.write("\n")
        f.close()

        numpy.savetxt( os.path.join(dir,"trans.txt") , self.trans_prob , delimiter="\t" )
        numpy.savetxt( os.path.join(dir,"trans_bos.txt") , self.trans_prob_bos , delimiter="\t" )
        numpy.savetxt( os.path.join(dir,"trans_eos.txt") , self.trans_prob_eos , delimiter="\t" )


def main():
    segm = HSMMWordSegm( 3 )
    segm.load_data( "data.txt" )
    segm.print_result()

    for it in range(100):
        print it
        segm.learn()
        print segm.num_vocab

    segm.learn( True )
    segm.save_result("result")
    return






if __name__ == '__main__':
    main()