import numpy as np
from utils import helper
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch
class l2():
    def __init__(self):
        self.name = 'l2_loss'

    def __call__(self,target,output):
        loss = (target - output)**2
        return loss.mean()

class meanf():
    def __init__(self):
        self.name = 'mean_frequency'
        self.freq = np.array([helper.index2hz(i) for i in range(helper.resolution)])

    def __call__(self,target,output):
        deltas = 0
        for x,y in zip(target,output):
            f1 = (x*self.freq).sum()/x.sum()
            f2 = (y*self.freq).sum()/y.sum()
            delta = abs(f2-f1)/f1
            deltas += delta
        return deltas / len(target)


class kmeans():
    def __init__(self,target):
        target = target.reshape(-1,target.shape[-1])
        self.name = 'kmeans'
        self.kmeans = KMeans(n_clusters=4).fit(target)
        self.save_center('kmeans')

    def save_center(self, filename):
        centers = self.kmeans.cluster_centers_
        np.save(filename, centers)

    def __call__(self, target, output):
        label1 = self.kmeans.predict(target)
        label2 = self.kmeans.predict(output)
        precise_overall = 0
        for i in range(self.kmeans.n_clusters):
            correct_num = sum((label1 == label2) & (label1 == i))
            all_num = sum(label1 == i)
            precise = correct_num / all_num
            precise_overall += precise
        return precise_overall/self.kmeans.n_clusters

class score_():
    def __init__(self,band_width = 4):
        self.name = 'our_metric'
        self.band_width = band_width

    def diff_band_point2set(self,p,i,lst):
        length = len(lst)
        dist_min = 1
        for d in range(-self.band_width, self.band_width+1):
            j = i+d
            if j >= 0 and j < length:
                dist_min = min( 
                                (d/self.band_width)**2 + (p - lst[j])**2,
                                dist_min
                            )
        return dist_min
    
    def diff_band_set2set(self, lst1, lst2):
        dist_all = 0
        for i,p in enumerate(lst1):
            dist_all += self.diff_band_point2set(p,i,lst2)
        for i,p in enumerate(lst2):
            dist_all += self.diff_band_point2set(p,i,lst1)
        return dist_all / (len(lst1) + len(lst1))

    def __call__(self, target, output):
        score = 0
        for x,y in zip(target,output):
            score += self.diff_band_set2set(x,y)
        return score / len(target)

class score():
    def __init__(self,band_width = 4):
        self.name = 'our_metric'
        self.band_width = band_width

    def batch_call(self, target, output):
        target = torch.tensor(target).cuda()
        output = torch.tensor(output).cuda()
        t2o = None
        o2t = None
        for d in range(-self.band_width, self.band_width+1):
            t = F.pad(target,(-d,d),value=100)
            o = F.pad(output,(-d,d),value=100)
            t2o_ = (target-o)**2 + (d/self.band_width)**2
            o2t_ = (output-t)**2 + (d/self.band_width)**2
            if t2o is None:
                t2o = t2o_
            else:
                t2o = torch.stack([t2o,t2o_],dim = -1).min(-1)[0]
            if o2t is None:
                o2t = o2t_  
            else:
                o2t = torch.stack([o2t,o2t_],dim = -1).min(-1)[0]
        t2o = t2o.mean(-1)
        o2t = o2t.mean(-1)
        dist = (t2o + o2t)/2
        return dist.mean().item()

    def __call__(self, target_, output_):
        score = 0
        step = 20
        step_length = len(target_) // step
        for i in range(step):
            target = target_[i*step_length:(i+1)*step_length]
            output = output_[i*step_length:(i+1)*step_length]
            score += self.batch_call(target,output)
        
        if step*step_length < len(target_):
            target = target_[step*step_length:]
            output = output_[step*step_length:]  
            score += self.batch_call(target,output)
            step += 1
        
        return score / step


    