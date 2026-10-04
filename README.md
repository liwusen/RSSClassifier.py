# RSS 分类器

基于朴素贝叶斯的中文 RSS 文章标题分类器，使用 jieba 分词。

## 分类

- `techblog` — 科技博客
- `news` — 新闻
- `special` — 精选内容
- `finance` — 财经
- `science` — 科普

## 使用

```bash
pip install -r requirements.txt
python main.py
```

训练数据保存在 `rss.json`，已训练标题记录在 `classifierTrained.json`。

## 自动化

GitHub Actions 每日 UTC 00:00 自动训练并提交更新。
