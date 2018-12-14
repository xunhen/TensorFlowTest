import numpy as np


class dataset(object):

    def __init__(self, filename, class_num=5, channel=16, train_ratio=0.6, val_ratio=0.2, window=300, windowshift=50,
                 data_start_per=0,
                 data_length_per=None,
                 start_line=3):
        self._filename = filename

        self._channel = channel
        self._class_num = class_num
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio

        self._window = window
        self._windowshift = windowshift

        self._data_start_per = data_start_per
        self._data_length_per = data_length_per

        self._start_line = start_line

        self._data_split = dict()
        self._data = np.array([])
        self._label = np.array([])

        self.data_style = ['train', 'test', 'val', 'trainval']
        self._cur = {i: None for i in self.data_style}
        self._perm = {i: None for i in self.data_style}

    def _get_data(self, filename, start_line=3):
        data = []
        with open(filename) as file:
            for line in file.readlines():
                if start_line > 0:
                    start_line -= 1
                    continue
                temp = np.array([float(i) for i in line.split()])
                data.append(temp[:self._channel])
        data = np.array(data)
        num_per_class = np.size(data, 0) // self._class_num

        start = self._data_start_per
        if not self._data_length_per or start + self._data_length_per > num_per_class:
            length = num_per_class - start
        else:
            length = self._data_length_per

        out_length = (length - self._window) // self._windowshift + 1

        self._label = np.ones([out_length * self._class_num]) * (
            np.array([[i] * out_length for i in range(self._class_num)]).reshape([-1]))
        self._data = []

        for id in range(self._class_num):
            _start = id * num_per_class + start
            _end = _start + length
            while _start + self._windowshift <= _end:
                temp = data[_start:_start + self._window, :]
                # temp = np.sqrt(np.mean(temp, axis=0, dtype=np.float64))
                # temp = np.mean(temp, axis=0, dtype=np.float64)
                self._data.append(temp)
                _start += self._windowshift

        self._data = np.array(self._data)
        self._num_per_class = out_length
        return self._data

    def _get_train_val_test_data(self):
        self.num_train_per_class = int(self.num_per_class * self._train_ratio)
        self.num_val_per_class = int(self.num_per_class * self._val_ratio)
        self.num_trainval_per_class = self.num_train_per_class + self.num_val_per_class
        self.num_test_per_class = self.num_per_class - self.num_trainval_per_class

        self.train_data = []
        self.train_label = []
        self.val_data = []
        self.val_label = []
        self.test_data = []
        self.test_label = []
        for id in range(self._class_num):
            start = id * self.num_per_class
            self.train_data.append(self._getDataByLength(self.data, start, self.num_train_per_class))
            self.train_label.append(self._getDataByLength(self.label, start, self.num_train_per_class))
            self.val_data.append(
                self._getDataByLength(self.data, start + self.num_train_per_class, self.num_val_per_class))
            self.val_label.append(
                self._getDataByLength(self.label, start + self.num_train_per_class, self.num_val_per_class))
            self.test_data.append(
                self._getDataByLength(self.data, start + self.num_train_per_class + self.num_val_per_class,
                                      self.num_test_per_class))
            self.test_label.append(
                self._getDataByLength(self.label, start + self.num_train_per_class + self.num_val_per_class,
                                      self.num_test_per_class))
        self._data_split = {
            'train_data': np.array(self.train_data).reshape([-1]),
            'train_label': np.array(self.train_label).reshape([-1]),
            'val_data': np.array(self.val_data).reshape([-1]),
            'val_label': np.array(self.val_label).reshape([-1]),
            'test_data': np.array(self.test_data).reshape([-1]),
            'test_label': np.array(self.test_label).reshape([-1])
        }

        return self._data_split

    def _getDataByLength(self, data, start, length):
        return data[start:start + length]

    def _shuffle(self, length, shuffle, style):
        if shuffle:
            self._perm[style] = np.random.permutation(np.arange(length))
        else:
            self._perm[style] = np.arange(length)
        self._cur[style] = 0

    def get_next_minibatch(self, batch_size=256, style='train', reset=False, is_shuffle=True):
        """
            get_next_minibatch(self, style='train')

                Parameters
                ----------
                reset : bool
                    if True : the state would be reset by using _shuffle()

                style : string, 'train','val','test','trainval'
                    default is 'train'
                batch_size : int,

                Returns
                -------
                the next minibatch about data and label
            """
        length = len(self.data_split[style + '_data'])
        if reset or self._cur[style] is None or self._cur[style] + batch_size > length:
            self._shuffle(length, is_shuffle,style)
        cur = self._cur[style]
        self._cur[style] += batch_size
        # print('self._perm[cur:cur + batch_size]', self._perm[cur:cur + batch_size])
        return self.data_split[style + '_data'][self._perm[style]][cur:cur + batch_size], \
               self.data_split[style + '_label'][self._perm[style]][cur:cur + batch_size]

    @property
    def data(self):
        if not len(self._data):
            self._get_data(self._filename, self._start_line)
        return self._data

    @property
    def label(self):
        if not len(self._label):
            self._get_data(self._filename, self._start_line)
        return self._label

    @property
    def data_split(self):
        if not len(self._data_split):
            self._get_train_val_test_data()
        return self._data_split

    def get_length(self,which):
        return eval('self.'+which+'_data_length')

    @property
    def train_data_length(self):
        return self.num_train_per_class*self._class_num

    @property
    def val_data_length(self):
        return self.num_val_per_class * self._class_num

    @property
    def test_data_length(self):
        return self.num_test_per_class * self._class_num

    @property
    def trainval_data_length(self):
        return self.num_trainval_per_class * self._class_num

    @property
    def num_per_class(self):
        if not len(self._data):
            self._get_data(self._filename, self._start_line)
        return self._num_per_class


if __name__ == '__main__':
    filename = 'LS-0-1-3-8-9.txt'
    _dataset = dataset(filename)
    print('len(_dataset.data) :', len(_dataset.data))
    print('_dataset.data[0] :', _dataset.data[0])
    print('_dataset.get_minibatch(reset=True)', _dataset.get_next_minibatch(reset=True))

    print('_dataset.get_minibatch()', _dataset.get_next_minibatch())
