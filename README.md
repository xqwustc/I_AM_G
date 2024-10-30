   # I_AM_G
   The implementation of Interest Augmented Multimodal Generator for Item Personalization.
   
   ## Dataset Preparation
   1. **Movielens**  
      - A popular dataset for movie recommendations, containing user ratings and movie information.  
      - Link: [Movielens Dataset](https://grouplens.org/datasets/movielens/)
   
   2. **POG**  
      - A dataset designed for outfit.  
      - Link: [POG Dataset](https://github.com/wenyuer/POG)
   
   3. **MIND**  
      - A large-scale dataset for news recommendation, containing user interactions and news articles.  
      - Link: [MIND News Dataset](https://www.kaggle.com/datasets/arashnic/mind-news-dataset)
   
   
   
   ## Training Steps
   
   1. Use the `/code/model/foreground/get_foreground.py` and try to extract the image foreground of each original image.
   2. Use LLaMA and LLaVA to obtain text and image tags.
   3. Format the data according to the dataset's required structure with `/code/data/data_processor.py`.
   4. Train across the pipeline:
      ```bash
      cd code
      ./train.sh
       ```
   
   ### Acknowledgments
   
   Our work heavily relies on the excellent contributions of [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter). We sincerely thank the team for their efforts.
   
   
   
