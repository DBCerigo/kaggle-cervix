import cv2

def _get_grayscale_img(path, rescale_dim):
    img = cv2.imread(path)
    rescaled = cv2.resize(img, (rescale_dim, rescale_dim), cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rescaled, cv2.COLOR_RGB2GRAY).astype('float')
    return gray

def save_img(img, num_id, desc):
    path = '../data/'+desc+'/'+num_id+".jpg"
    cv2.imwrite(path,img)
    return path
    
def _save_grayscale_row(row, desc):
    path = row.path
    num_id = row.num_id
    gray = _get_grayscale_img(path, 100)
    return save_img(gray, num_id, desc)

def save_grayscale(df):
    for index, row in df.iterrows():
        df.loc[index,'gray_path'] = _save_grayscale_row(row,desc)
    return df

def random_forest_transform(df, test=False):
    forest_df = pd.DataFrame()
    vecs = []
    types = []
    ids = []
    for _, row in df.iterrows():
        gray = cv2.imread(row.gray_path)
        vec = process_image(gray)
        if not test:
            cervix_type = row.type
            types.append(cervix_type[-1])
        else:
            ids.append(row.num_id)
        vecs.append(vec)
    if not test:
        return np.squeeze(np.array(vecs)), np.array(types)
    else:
        return np.squeeze(np.array(vecs)), np.array(ids)
                              
def process_image(img):
    """Normalize image and turn into array of one long column"""
    normalized = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    vec = normalized.reshape(1, np.prod(normalized.shape))
    return vec / np.linalg.norm(vec)
