import helper

data_dir = './data'

def download():
    helper.download_extract('mnist', data_dir)
    helper.download_extract('celeba', data_dir)
