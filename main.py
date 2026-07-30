import jieba
import feedparser, os
import math, time, json, pickle


def print_all_var():
    for i in globals().keys():
        if (not i.startswith("__")) and (not i.endswith("__")):
            print(i)


def variance(vals):
    mean = float(sum(vals)) / len(vals)
    s = sum([(x - mean) ** 2 for x in vals])
    return s / len(vals)


def _migrate_pickle_to_json(pickle_path, json_path):
    if os.path.exists(pickle_path) and not os.path.exists(json_path):
        print(f"[Migrate] Converting {pickle_path} -> {json_path}")
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    return None


class classifier:
    def __init__(self, getfeatures, filename=None, trainedfilename=None):
        self.fc = {}
        self.cc = {}
        self.getfeatures = getfeatures
        self.thresholds = {}
        self.filename = filename
        self.trainedfilename = trainedfilename

        if filename is not None:
            _migrate_pickle_to_json(
                filename.replace(".json", ".dat"), filename
            )
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    self.cc, self.fc = json.load(f)

        self.trained = []
        if trainedfilename is not None:
            _migrate_pickle_to_json(
                trainedfilename.replace(".json", ".dat"), trainedfilename
            )
            if os.path.exists(trainedfilename):
                with open(trainedfilename, "r", encoding="utf-8") as f:
                    self.trained = json.load(f)

    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    def totalcount(self):
        return sum(self.cc.values())

    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.getfeatures(item)
        for f in features:
            self.incf(f, cat)
        self.incc(cat)

    def saveToFile(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump([self.cc, self.fc], f, ensure_ascii=False, indent=2)
        with open(self.trainedfilename, "w", encoding="utf-8") as f:
            json.dump(self.trained, f, ensure_ascii=False, indent=2)
        print("[Save] Saved to", self.filename)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0
        return self.fcount(f, cat) / self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        basicprob = prf(f, cat)
        totals = sum([self.fcount(f, c) for c in self.categories()])
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp

    def trainsAllFromRss(self, url, tpe):
        sumT = 0
        startt = time.time()
        f = feedparser.parse(url)
        for entry in f["entries"]:
            if entry["title"] not in self.trained:
                self.train(entry["title"], tpe)
                self.trained.append(entry["title"])
                sumT += 1
        print("[", url, "] Success:\t", "time=", time.time() - startt, "S")
        return sumT

    def classify(self, f):
        k = self.cc.keys()
        fs = list(jieba.cut(f))
        if not fs:
            return dict.fromkeys(k, 0.0)
        prog = {}
        for key in k:
            summ = 0
            for word in fs:
                summ += self.fprob(word, key)
            prog[key] = float(summ / len(fs))
        return prog

    def pruning(self, alpha=0.1, delta=5):
        sumPrun = 0
        keys = list(self.fc.keys())
        willPop = []
        print("[Pruning] Start")
        for key in keys:
            probs = []
            for tpe in self.cc.keys():
                probs.append(self.fprob(key, tpe))
            if variance(probs) < alpha and sum(self.fc[key].values()) > delta:
                willPop.append(key)
                print("\t->", key)
                sumPrun += 1
        for key in willPop:
            del self.fc[key]
        print("[Pruning] Deleted Count=", sumPrun)

    def mostMentionedWords(self, n: int = 10):
        word_counts = {word: sum(self.fc[word].values()) for word in self.fc}
        most_mentioned = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        return most_mentioned

if __name__ == "__main__":
    print("[Main] Ready")
    print(list(jieba.cut("测试")))
    print("[Main] Jieba Ready")

    print("[Main] Start")
    cl = classifier(
        jieba.cut, filename="rss.json", trainedfilename="classifierTrained.json"
    )
    traningUrls = {
        # techblog
        "https://www.labno3.com/feed/": "techblog",
        "https://www.ruanyifeng.com/blog/atom.xml": "techblog",
        "https://feed.cnblogs.com/blog/sitehome/rss": "techblog",
        "https://sspai.com/feed": "techblog",
        "https://www.ifanr.com/feed": "techblog",
        "https://www.oschina.net/news/rss": "techblog",
        "https://rss.aishort.top/?type=juejin": "techblog",
        "https://www.solidot.org/index.rss": "techblog",
        # special
        "https://rss.aishort.top/?type=zhihu": "special",
        "https://rss.aishort.top/?type=guokr": "special",
        # news
        "https://rss.aishort.top/?type=baidu": "news",
        "http://www.people.com.cn/rss/ywkx.xml": "news",
        "https://feed.cnblogs.com/news/rss": "news",
        "https://rss.huxiu.com/": "news",
        "https://www.techweb.com.cn/rss/allnews.xml": "news",
        "http://www.xinhuanet.com/politics/xhll.xml": "news",
        "https://rss.aishort.top/?type=toutiao": "news",
        "https://rss.aishort.top/?type=163": "news",
        "https://feedx.net/rss/weibo.xml": "news",
        # finance
        "https://rss.aishort.top/?type=36kr": "finance",
        "https://rss.aishort.top/?type=cls": "finance",
        "https://rss.aishort.top/?type=wallstreetcn": "finance",
        "https://rss.aishort.top/?type=eastmoney": "finance",
        "https://rss.aishort.top/?type=caijing": "finance",
        # science
        "https://rss.aishort.top/?type=scicat": "science",
        "https://rss.aishort.top/?type=kepu": "science",
        "https://rss.aishort.top/?type=cas": "science",
        "https://rss.aishort.top/?type=zhishexiao": "science",
    }
    TrainsCount = 0
    for url in traningUrls.keys():
        if not url.startswith("#"):
            TrainsCount += cl.trainsAllFromRss(url, traningUrls[url])
        else:
            print("[", url, "] Skiped")
        if url.startswith("!"):
            break
#    cl.pruning(alpha=0.03)
    cl.saveToFile()
    print(
        "[Main] 总训练：",
        TrainsCount,
        "数据总量：",
        len(cl.fc.keys()),
        "文件大小：",
        str(float(os.path.getsize("rss.json") / 1024)) + "KB",
    )
    print("[Main] 最常出现的词：", cl.mostMentionedWords(10))
    print("[Main] 测试分类")
    testUrls = {
        "https://www.ruanyifeng.com/blog/atom.xml": "techblog",
        "https://rss.aishort.top/?type=zhihu": "special",
        "https://www.techweb.com.cn/rss/allnews.xml": "news",
        "https://rss.aishort.top/?type=36kr": "finance",
        "https://rss.aishort.top/?type=scicat": "science",
    }
    correct = 0
    summ = 0
    for url in testUrls.keys():
        if not url.startswith("#"):
            f = feedparser.parse(url)
            for entry in f["entries"]:
                result = cl.classify(entry["title"])
                summ += 1
                if testUrls[url] == max(result, key=result.get):
                    correct += 1
            print(f"[Main] 测试完毕:{url}")
        else:
            print("[", url, "] Skiped")
    print("[Main] 测试分类完毕")
    print(
        "[Main] 正确率：",
        correct,
        "/",
        summ,
        "=",
        round(correct / (summ + 0.00001), 2),
    )
