from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import os
import random
import util.util as util
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class FtranslationDataset(BaseDataset):
    """ Датасет для визуального перевода шрифта.
		Требует 3 папок: source (картинки-скелеты букв); 
		язык, с которого переводится шрифт(style language); 
		язык, на который переводим шрифт (content language).
	"""
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--style_channel', type=int, default=6, help='# of style channels')        
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.style_language, self.content_language = opt.direction.split('2')

        self.dataroot = opt.dataroot
        # get the image directory    
        if (opt.phase == 'generate'):
            self.source_dir = os.path.join(opt.dataroot, opt.phase, 'source', self.content_language) 
        else:
            self.source_dir = os.path.join(self.dataroot, 'source') 
        self.content_dir = os.path.join(self.dataroot, self.content_language)
        self.paths = sorted(make_dataset(self.content_dir, opt.max_dataset_size))

        self.style_channel = opt.style_channel
        #self.img_size = opt.crop_size
        self.to_tensor = self.get_tr_image2tensor(opt)
		
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains ?????
        """
        gt_path = self.paths[index]
        style_paths = self.get_style_paths(gt_path)
        content_path = self.get_content_path(gt_path)
        
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        if self.opt.phase == 'train':
            is_gray = ('grayscale' in self.opt.preprocess) or (self.opt.input_nc == 1)
            transform = get_transform(modified_opt, grayscale=is_gray)
        else:
            transform = self.to_tensor
        
        source_image = self.load_image(content_path, self.to_tensor)
        gt_image = self.load_image(gt_path, self.to_tensor)

        if self.opt.phase == "train":
            image_paths = gt_path
        else:
            fontname = style_paths[0].split(os.sep)[-2]
            parts = gt_path.split(os.sep)
            image_paths = os.path.join(fontname, parts[-1])

        style_image = torch.cat([self.load_image(style_path, transform) for style_path in style_paths], 0)
        # собираем в одно изображение, чтобы было понятно, какую букву генерировать
        style_image = torch.cat([style_image, source_image], 0)
        #todo
        return {'style_images':style_image, 'gt_images':gt_image, 'style_paths':style_paths, 'image_paths':image_paths}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)  

    def load_image(self, path, transform):
        """ Загружает картинку и трансформирует её согласно transform.
            Parameters:
                path (string) - полный путь до картинки
                transform (torchvision.transforms.Compose) - нужные преобразования
            Returns:
                результат трансформаций transform (torch.Tensor или PIL.Image)
        """
        image = Image.open(path).convert('RGB')
        tensor_image = transform(image)
        image.close()
        return tensor_image    

    def get_tr_image2tensor(self, opt):
        """ Возвращает преобразования без аугментации. В преобразования обязательно
            входит трансформация в тензор и нормализация. Если --input_nc 1, то 
            ещё перевод в градации серого.
            Parameters:
                opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            Returns:
                собранные трансформации в виде объекта torchvision.transforms.Compose
        """
        ts = []
        if opt.input_nc == 1:
            ts.append(transforms.Grayscale(1))
        if 'crop' in opt.preprocess:
            to_size = opt.crop_size
        else:
            to_size = opt.load_size
        ts.append(transforms.Resize(to_size, Image.BICUBIC))
        ts += [transforms.ToTensor(), transforms.Normalize(mean = (0.5), std = (0.5))]
        return transforms.Compose(ts)

    def get_style_paths(self, path):
        """Возвращает пути до стилевых изображений того же шрифта.
            Parameters:
                path (string) - полный путь до картинки, включает в себя шрифт как последнюю папку
            Returns:
                список абсолютных путей до стил. изображений того же шрифта со случайно 
                выбранными буквами; длина списка = self.style_channel
        """
        parts = path.split(os.sep)
        # путь идет как .../папка_шрифта/буква.png
        font_name = parts[-2]
        target_font_dir = os.path.join(self.dataroot, self.style_language, font_name)
        style_letters = [os.path.join(target_font_dir, letter) for letter in random.sample(os.listdir(target_font_dir), self.style_channel)]
        return style_letters

    def get_content_path(self, path):
        """Возвращает source-картинку этой буквы.
            Parameters:
                path (string) - полный путь до картинки символа
            Returns:
                полный путь до source-картинки того же символа, что и в переданном пути
        """
        letter = path.split(os.sep)[-1]
        return os.path.join(self.source_dir, letter)
