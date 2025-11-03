#!/usr/bin/env python
# coding: utf-8

# In[5]:
if __name__ == "__main__":

    from fastbook import *
    from ddgs import DDGS
    from pathlib import Path
    from fastdownload import download_url

    def search_images_ddg(keywords, max_images=200):
        """Search images using DuckDuckGo's ddgs library"""
        with DDGS() as ddgs:
            results = ddgs.images(keywords, max_results=max_images)
            return L(results).itemgot("image")

    # Example usage:
    urls = search_images_ddg('bird photos', max_images=1)

    if len(urls) == 0:
        print("No images found!")
    else:
        print(f"Number of URLs found: {len(urls)}")
        print(f"First image URL: {urls[0]}")


    # In[6]:


    dest = Path('bird.jpg')
    if len(urls) > 0 and not dest.exists():
        download_url(urls[0], dest, show_progress=False)


    # In[7]:


    im = Image.open(dest)
    im.to_thumb(256,256)


    # In[8]:


    searches = 'forest', 'bird'
    path = Path('bird_or_not')

    if not path.exists():
        for o in searches:
            dest = (path/o)
            dest.mkdir(parents=True, exist_ok=True)
            results = search_images_ddg(f'{o} photo')
            download_images(dest, urls=results[:200])
            resize_images(dest, max_size=400, dest=dest)



    # In[9]:


    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)    


    # In[10]:


    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')],
    ).dataloaders(path, bs=64)
    dls.show_batch(max_n=12)


    # In[11]:


    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(8)


    # In[14]:


    is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
    class_idx = learn.dls.vocab.o2i[is_bird]  # maps predicted class → index
    prob = probs[class_idx]
    print(f"This is a: {is_bird}. Probability: {prob:.4f}")


    # In[15]:


    download_url(search_images_ddg('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
    Image.open('forest.jpg').to_thumb(256,256)


    # In[38]:


    is_bird, _, probs = learn.predict(PILImage.create('forest.jpg'))
    class_idx = learn.dls.vocab.o2i[is_bird]
    prob = probs[class_idx]
    print(f"This is a: {is_bird}. Probability: {prob:.4f}")


    # In[40]:


    download_url(search_images_ddg('forest photos', max_images=1)[0], 'forest2.jpg', show_progress=False)
    Image.open('forest.jpg').to_thumb(256,256)


    # In[41]:


    is_bird, _, probs = learn.predict(PILImage.create('forest2.jpg'))
    class_idx = learn.dls.vocab.o2i[is_bird]
    prob = probs[class_idx]
    print(f"This is a: {is_bird}. Probability: {prob:.4f}")


    # In[ ]:




