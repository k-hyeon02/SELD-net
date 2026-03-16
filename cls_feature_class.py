# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav  # wav 파일 읽기
import utils
from sklearn import preprocessing  # StandardScaler
from sklearn.externals import joblib  # scaler 저장/불러오기
from IPython import embed  # 디버깅
import matplotlib.pyplot as plot
plot.switch_backend('agg')  # 디스플레이 없는 서버에서 그래프 저장 가능하게


class FeatureClass:
    '''
    dataset : 어떤 데이터셋인지 (ansim, resim 등)
    ov : 최대 동시 겹치는 소리 수 (1, 2, 3)
    split : 교차검증 분할 번호
    nfft : FFT 크기 (주파수 해상도 결정)
    db : 신호 대 잡음비 (30dB 고정)
    wav_extra_name, desc_extra_name : 경로 뒤에 붙이는 접미사 (특수 실험용)
    '''
    def __init__(self, dataset='ansim', ov=3, split=1, nfft=1024, db=30, wav_extra_name='', desc_extra_name=''):

        _data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        if dataset == 'ansim':
            self._base_folder = os.path.join(_data_root, 'ansim')
        elif dataset == 'resim':
            self._base_folder = os.path.join(_data_root, 'resim')
        elif dataset == 'cansim':
            self._base_folder = os.path.join(_data_root, 'cansim')
        elif dataset == 'cresim':
            self._base_folder = os.path.join(_data_root, 'cresim')
        elif dataset == 'real':
            self._base_folder = os.path.join(_data_root, 'real')
        elif dataset == 'mansim':
            self._base_folder = os.path.join(_data_root, 'mansim')
        elif dataset == 'mreal':
            self._base_folder = os.path.join(_data_root, 'mreal')

        # Input directories
        self._aud_dir = os.path.join(self._base_folder, 'wav_ov{}_split{}_{}db{}'.format(ov, split, db, wav_extra_name))  # 오디오 wav 파일이 있는 폴더
        self._desc_dir = os.path.join(self._base_folder, 'desc_ov{}_split{}{}'.format(ov, split, desc_extra_name))  # 각 오디오에 대응하는 csv 메타데이터가 있는 폴더

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._mode = None
        self._ov = ov
        self._split = split
        self._db = db
        self._nfft = nfft
        self._win_len = self._nfft  # FFT window 크기
        self._hop_len = self._nfft//2  # widow의 50% overlap
        self._dataset = dataset
        self._eps = np.spacing(np.float(1e-16))  # epsilon : 0으로 나누기 방지

        # If circular-array 8 channels else 4 for Ambisonic
        if 'c' in self._dataset:
            self._nb_channels = 8  # 원형 마이크 배열 (8채널)
        else:
            self._nb_channels = 4  # Ambisonic (4채널 : W,X,Y,Z)

        # Sound event classes dictionary
        self._unique_classes = dict()
        if 'real' in self._dataset:
            # Urbansound8k sound events
            self._unique_classes = \
                {
                    '1': 0,
                    '3': 1,
                    '4': 2,
                    '5': 3,
                    '6': 4,
                    '7': 5,
                    '8': 6,
                    '9': 7
                }
        else:
            # DCASE 2016 Task 2 sound events
            self._unique_classes = \
                {
                    'clearthroat': 2,
                    'cough': 8,
                    'doorslam': 9,
                    'drawer': 1,
                    'keyboard': 6,
                    'keysDrop': 4,
                    'knock': 0,
                    'laughter': 10,
                    'pageturn': 7,
                    'phone': 3,
                    'speech': 5
                }

        self._fs = 44100  # sampling rate = 44.1kHz
        self._frame_res = self._fs / float(self._hop_len)  # 1초당 프레임 수 = 44100/256 = 172
        self._hop_len_s = self._nfft/2.0/self._fs  # hop_len 초 단위 = 256/44100=0.0058s
        self._nb_frames_1s = int(1 / self._hop_len_s)  # 1s=172 frame
        self._fade_win_size = 0.01 * self._fs  # 10ms = 441 samples

        # DOA estimation에 사용할 azimuth/elevation 그리드 (10도 간격)
        self._resolution = 10  # 방위각/고도각 해상도 = 10도
        self._azi_list = range(-180, 180, self._resolution)  # azimuth -180~180도 10도 간격으로 - 36개
        self._length = len(self._azi_list)  # 36
        self._ele_list = range(-60, 60, self._resolution) # elevation -60~60도 10도 간격으로 - 12개
        self._height = len(self._ele_list)  # 12
        self._weakness = None

        # For regression task only: 소리가 없는 구간의 레이블 기본값 - 그리드 밖의 값
        self._default_azi = 180
        self._default_ele = 60

        # 기본값이 실제 그리드 안에 포함되면 종료 (안전장치)
        if self._default_azi in self._azi_list:
            print('ERROR: chosen default_azi value {} should not exist in azi_list'.format(self._default_azi))
            exit()
        if self._default_ele in self._ele_list:
            print('ERROR: chosen default_ele value {} should not exist in ele_list'.format(self._default_ele))
            exit()

        self._audio_max_len_samples = 30 * self._fs  # 모든 오디오를 30초로 통일 30s * 44100 = 1,323,000 samples
        # TODO: Fix the audio synthesis code to always generate 30s of
        # audio. Currently it generates audio till the last active sound event, which is not always 30s long. This is a
        # quick fix to overcome that. We need this because, for processing and training we need the length of features
        # to be fixed.

        self._max_frames = int(np.ceil((self._audio_max_len_samples - self._win_len) / float(self._hop_len)))
        # 30초 오디오에서 FFT 슬라이딩하면 나오는 프레임 수 : (1,323,000-512)/256=5166 프레임

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)  # (샘플수, 채널수)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps  # 필요한 채널수만큼 자르고 
        # 16bit wav 파일의 샘플값(-32768~32767) : int16 -> float32로 정규화 -1.0~1.0
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.zeros((self._audio_max_len_samples - audio.shape[0], audio.shape[1]))  
            audio = np.vstack((audio, zero_pad))  # 30초보다 짧으면 뒤에 제로패딩
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]  # 30초보다 길면 잘라냄
        return audio, fs  # audio shape : (1,323,000, 채널수), fs : 44100

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()  # x보다 크거나 같은 2의 거듭제곱 반환

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]  # 채널 수
        hann_win = np.repeat(np.hanning(self._win_len)[np.newaxis].T, _nb_ch, 1)  # 모든 채널에 동일한 윈도우를 한번에 곱하기 위함
        nb_bins = self._nfft // 2  # 256
        spectra = np.zeros((self._max_frames, nb_bins, _nb_ch), dtype=complex)  # 결과 저장 배열 초기화, 복소수로 저장(크기+위상)
        
        for ind in range(self._max_frames): # ind : 0, 1, 2, ... , 5156
            start_ind = ind * self._hop_len  # 프레임의 시작 샘플 위치 (256씩 이동) 0, 256, 512, ...
            aud_frame = audio_input[start_ind + np.arange(0, self._win_len), :] * hann_win  # 윈도잉
            spectra[ind] = np.fft.fft(aud_frame, n=self._nfft, axis=0, norm='ortho')[:nb_bins, :]

        return spectra
        # 최종 출력 shape : (max_frames, 256, 4) 
        # max_frames: 시간 프레임 수 (5166), 256: 주파수 bin 수, 4: 채널 수, dtype: complex(크기+위상 보존) 

    def _extract_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(os.path.join(self._aud_dir, audio_filename))
        audio_spec = self._spectrogram(audio_in)
        print(audio_spec.shape)  # (5166, 256, 4) : 시간 프레임 수, 주파수 bin 수, 채널 수
        # (5166, 256, 4) -> (5166, 1024)
        np.save(os.path.join(self._feat_dir, audio_filename), audio_spec.reshape(self._max_frames, -1))

    # OUTPUT LABELS
    def _read_desc_file(self, desc_filename):
        # csv 메타데이터 파일 내용 담을 빈 딕셔너리 생성
        desc_file = {
            'class': list(), 'start': list(), 'end': list(), 'ele': list(), 'azi': list(),
            'ele_dir': list(), 'azi_dir': list(), 'ang_vel': list(), 'dist': list()
        }
        fid = open(os.path.join(self._desc_dir, desc_filename), 'r')
        next(fid)  # 헤더 행 스킵

        for line in fid:
            split_line = line.strip().split(',')  # csv 각 행을 쉼표로 분리
            # class명 추출
            if 'real' in self._dataset:  # real dataset인 경우 
                desc_file['class'].append(split_line[0].split('.')[0].split('-')[1])
            else:
                desc_file['class'].append(split_line[0].split('.')[0][:-3])

            desc_file['start'].append(int(np.floor(float(split_line[1])*self._frame_res)))
            desc_file['end'].append(int(np.ceil(float(split_line[2])*self._frame_res)))
            desc_file['ele'].append(int(float(split_line[3])))
            desc_file['azi'].append(int(float(split_line[4])))

            if self._dataset[0] is 'm':  # moving sound source dataset인 경우
                if 'real' in self._dataset:
                    desc_file['ang_vel'].append(int(float(split_line[5])))
                    desc_file['dist'].append(float(split_line[6]))
                else:
                    desc_file['ele_dir'].append(int(float(split_line[5])))
                    desc_file['azi_dir'].append(int(float(split_line[6])))
                    desc_file['ang_vel'].append(int(float(split_line[7])))
                    desc_file['dist'].append(float(split_line[8]))
            else:
                desc_file['dist'].append(float(split_line[5]))
        fid.close()
        return desc_file

    def get_list_index(self, azi, ele):
        azi = (azi - self._azi_list[0]) // 10  # azimuth를 0~35 사이의 정수 인덱스로 반환
        ele = (ele - self._ele_list[0]) // 10  # elevation을 0~11 사이의 정수 인덱스로 반환
        return azi * self._height + ele  # 2D grid(azi, ele)를 1D 인덱스로 변환 (0~431 사이의 정수)

    def _get_matrix_index(self, ind):
        azi, ele = ind // self._height, ind % self._height  # 1D 인덱스를 2D grid(azi, ele)로 변환
        azi = (azi * 10 + self._azi_list[0])  # 인덱스를 실제 각도로 변환
        ele = (ele * 10 + self._ele_list[0])
        return azi, ele

    def get_vector_index(self, ind):  # 수평면(2D)만 고려하는 단순한 DOA 추정에 쓰이는 함수
        azi = (ind * 10 + self._azi_list[0])  # azimuth만 실제 각도로 변환
        return azi

    @staticmethod
    def scaled_cross_product(a, b):  # a -> b (unit vectors) 이동할 때 회전축 벡터
        ab = np.dot(a, b)  
        if ab > 1 or ab < -1:
            return [999]  # error

        acos_ab = np.arccos(ab)  # a와 b 사이의 각도(radian)
        x = np.cross(a, b)  # 외적 : a, b 둘 다에 수직인 벡터 = 회전축 방향

        if acos_ab == np.pi or acos_ab == 0 or sum(x) == 0:
            # a,b가 반대 방향(무한 개의 회전축), 같은 방향
            return [999]
        else:
            return x/np.sqrt(np.sum(x**2))  # 단위 벡터로 정규화된 회전축 벡터 반환

    def get_trajectory(self, event_length_s, _start_xyz, _rot_vec, _random_ang_vel):
        frames_per_sec = self._fs / self._fade_win_size  # 44100/441= 100frame/s : 초당 프레임
        ang_vel_per_win = _random_ang_vel / frames_per_sec  # 초당 각속도
        nb_frames = int(np.ceil(event_length_s * frames_per_sec))  # 이벤트 길이(초) -> 총 프레임 수
        xyz_array = np.zeros((nb_frames, 3))  # 결과 저장할 배열 : (총 프레임 수, xyz)
        for frame in range(nb_frames):
            _R = self.rotate_matrix_vec_ang(_rot_vec, frame * ang_vel_per_win)
            # ex) frame 0 → 0도 회전, frame 1 → 0.3도 회전, frame 2 → 0.6도 회전 ...
            xyz_array[frame, :] = np.dot(_start_xyz, _R.T)
            # 시작 위치를 R만큼 회전시켜 해당 프레임의 위치 계산

        return xyz_array


    @staticmethod
    def rotate_matrix_vec_ang(_rot_vec, theta):  # _rot_vec 방향으로 theta만큼 회전하는 회전 행렬 계산
        # u_x_u = u ⊗ u : 회전축의 외적 텐서 (3x3)
        u_x_u = np.array(
            [
                [_rot_vec[0] ** 2, _rot_vec[0] * _rot_vec[1], _rot_vec[0] * _rot_vec[2]],
                [_rot_vec[1] * _rot_vec[0], _rot_vec[1] ** 2, _rot_vec[1] * _rot_vec[2]],
                [_rot_vec[2] * _rot_vec[0], _rot_vec[2] * _rot_vec[1], _rot_vec[2] ** 2]
            ]
        )

        # u_x = [u]× : 회전축의 반대칭 행렬 (외적 연산자)
        u_x = np.array(
            [
                [0, -_rot_vec[2], _rot_vec[1]],
                [_rot_vec[2], 0, -_rot_vec[0]],
                [-_rot_vec[1], _rot_vec[0], 0]
            ]
        )
        # np.eye(3) * np.cos(theta)	: I·cos(θ) - 원래 방향 성분
        # np.sin(theta) * u_x : sin(θ)·[u]× - 회전축에 수직인 성분
        # (1-np.cos(theta)) * u_x_u	: (1-cos(θ))·(u ⊗ u) - 회전축 방향 성분
        return np.eye(3) * np.cos(theta) + np.sin(theta) * u_x + (1 - np.cos(theta)) * u_x_u

    @staticmethod
    def sph2cart(az, el, r):
        """
        Converts spherical coordinates given by azimuthal, elevation and radius to cartesian coordinates of x, y and z

        :param az: azimuth angle
        :param el: elevation angle
        :param r: radius
        :return: cartesian coordinate
        """
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    @staticmethod
    def cart2sph(x, y, z):
        XsqPlusYsq = x ** 2 + y ** 2
        r = np.sqrt(XsqPlusYsq + z ** 2)  # r
        elev = np.arctan2(z, np.sqrt(XsqPlusYsq))  # theta
        az = np.arctan2(y, x)  # phi
        return az, elev, r

    @staticmethod
    def wrapToPi(rad_list):
        xwrap = np.remainder(rad_list, 2 * np.pi)
        mask = np.abs(xwrap) > np.pi
        xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
        return xwrap

    def wrapTo180(self, deg_list):
        rad_list = deg_list * np.pi / 180.
        rad_list = self.wrapToPi(rad_list)
        deg_list = rad_list * 180 / np.pi
        return deg_list

    def _get_doa_labels_regr(self, _desc_file):
        azi_label = self._default_azi*np.ones((self._max_frames, len(self._unique_classes)))
        ele_label = self._default_ele*np.ones((self._max_frames, len(self._unique_classes)))
        for i, ele_ang in enumerate(_desc_file['ele']):
            start_frame = _desc_file['start'][i]
            if start_frame > self._max_frames:
                continue
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            nb_frames = end_frame - start_frame
            azi_ang = _desc_file['azi'][i]
            class_ind = self._unique_classes[_desc_file['class'][i]]
            if self._dataset[0] is 'm':
                if 'real' in self._dataset:
                    se_len_s = nb_frames / self._frame_res
                    azi_trajectory = np.floor(
                        np.linspace(azi_ang, azi_ang+_desc_file['ang_vel'][i]*se_len_s, nb_frames)
                    )
                    azi_ang = self.wrapTo180(azi_trajectory)

                else:
                    start_xyz = self.sph2cart(azi_ang*np.pi/180, ele_ang*np.pi/180, 1)
                    direction_xyz = self.sph2cart(_desc_file['azi_dir'][i]*np.pi/180, _desc_file['ele_dir'][i]*np.pi/180, 1)

                    rot_vec = self.scaled_cross_product(start_xyz, direction_xyz)
                    xyz_trajectory = self.get_trajectory(
                        nb_frames/self._frame_res, start_xyz, rot_vec, _desc_file['ang_vel'][i]*np.pi/180)

                    tmp_azi_ang, tmp_ele_ang, tmp_r = self.cart2sph(
                        xyz_trajectory[:, 0], xyz_trajectory[:, 1], xyz_trajectory[:, 2])
                    org_time = np.linspace(0, 1, tmp_azi_ang.shape[0])
                    new_time = np.linspace(0, 1, end_frame - start_frame)
                    azi_ang = np.interp(new_time, org_time, tmp_azi_ang * 180/np.pi)
                    ele_ang = np.interp(new_time, org_time, tmp_ele_ang * 180/np.pi)

            if np.sum(ele_ang >= self._ele_list[0]) and np.sum(ele_ang <= self._ele_list[-1]):
                azi_label[start_frame:end_frame, class_ind] = azi_ang
                ele_label[start_frame:end_frame, class_ind] = ele_ang
            else:
                # print(start_xyz, direction_xyz)
                print('bad_angle {} {}'.format(azi_ang, ele_ang))
        doa_label_regr = np.concatenate((azi_label, ele_label), axis=1)
        return doa_label_regr

    def _get_se_labels(self, _desc_file):
        se_label = np.zeros((self._max_frames, len(self._unique_classes)))
        for i, se_class in enumerate(_desc_file['class']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            se_label[start_frame:end_frame + 1, self._unique_classes[se_class]] = 1
        return se_label

    def _get_labels_for_file(self, label_filename, _desc_file):
        label_mat = None
        if self._mode is 'regr':
            se_label = self._get_se_labels(_desc_file)
            doa_label = self._get_doa_labels_regr(_desc_file)
            label_mat = np.concatenate((se_label, doa_label), axis=1)
        else:
            print("The supported modes are 'regr', you provided {}".format(self._mode))
        print(label_mat.shape)
        np.save(os.path.join(self._label_dir, label_filename), label_mat)

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self, extra=''):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir(extra)
        utils.create_folder(self._feat_dir)

        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            print('file_cnt {}, file_name {}'.format(file_cnt, file_name))
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            self._extract_spectrogram_for_file(wav_filename)

    def preprocess_features(self, extra=''):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir(extra)
        self._feat_dir_norm = self.get_normalized_feat_dir(extra)
        utils.create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file(extra)

        # pre-processing starts
        print('Estimating weights for normalizing feature files:')
        print('\t\tfeat_dir {}'.format(self._feat_dir))

        spec_scaler = preprocessing.StandardScaler()
        train_cnt = 0
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            if 'train' in file_name:
                print(file_cnt, train_cnt, file_name)
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(np.concatenate((np.abs(feat_file), np.angle(feat_file)), axis=1))
                del feat_file
                train_cnt += 1
        joblib.dump(
            spec_scaler,
            normalized_features_wts_file
        )

        print('Normalizing feature files:')
        # spec_scaler = joblib.load(normalized_features_wts_file) #load weights again using this command
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print(file_cnt, file_name)
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                feat_file = spec_scaler.transform(np.concatenate((np.abs(feat_file), np.angle(feat_file)), axis=1))
                np.save(
                    os.path.join(self._feat_dir_norm, file_name),
                    feat_file
                )
                del feat_file
        print('normalized files written to {} folder and the scaler to {}'.format(
            self._feat_dir_norm, normalized_features_wts_file))

    def normalize_features(self, extraname=''):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir(extraname)
        self._feat_dir_norm = self.get_normalized_feat_dir(extraname)
        utils.create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()

        # pre-processing starts
        print('Estimating weights for normalizing feature files:')
        print('\t\tfeat_dir {}'.format(self._feat_dir))

        spec_scaler = joblib.load(normalized_features_wts_file)
        print('Normalizing feature files:')
        # spec_scaler = joblib.load(normalized_features_wts_file) #load weights again using this command
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print(file_cnt, file_name)
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                feat_file = spec_scaler.transform(np.concatenate((np.abs(feat_file), np.angle(feat_file)), axis=1))
                np.save(
                    os.path.join(self._feat_dir_norm, file_name),
                    feat_file
                )
                del feat_file
        print('normalized files written to {} folder and the scaler to {}'.format(
            self._feat_dir_norm, normalized_features_wts_file))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self, mode='regr', weakness=0, extra=''):
        self._label_dir = self.get_label_dir(mode, weakness, extra)
        self._mode = mode
        self._weakness = weakness

        print('Extracting spectrogram and labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        utils.create_folder(self._label_dir)

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            print('file_cnt {}, file_name {}'.format(file_cnt, file_name))
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            desc_file = self._read_desc_file(file_name)
            self._get_labels_for_file(wav_filename, desc_file)

    # ------------------------------- Misc public functions -------------------------------
    def get_classes(self):
        return self._unique_classes

    def get_normalized_feat_dir(self, extra=''):
        return os.path.join(
            self._base_folder,
            'spec_ov{}_split{}_{}db_nfft{}{}_norm'.format(self._ov, self._split, self._db, self._nfft, extra)
        )

    def get_unnormalized_feat_dir(self, extra=''):
        return os.path.join(
            self._base_folder,
            'spec_ov{}_split{}_{}db_nfft{}{}'.format(self._ov, self._split, self._db, self._nfft, extra)
        )

    def get_label_dir(self, mode, weakness, extra=''):
        return os.path.join(
            self._base_folder,
            'label_ov{}_split{}_nfft{}_{}{}{}'.format(self._ov, self._split, self._nfft, mode, 0 if mode is 'regr' else weakness, extra)
        )

    def get_normalized_wts_file(self, extra=''):
        return os.path.join(
            self._base_folder,
            'spec_ov{}_split{}_{}db_nfft{}{}_wts'.format(self._ov, self._split, self._db, self._nfft, extra)
        )

    def get_default_azi_ele_regr(self):
        return self._default_azi, self._default_ele

    def get_nb_channels(self):
        return self._nb_channels

    def nb_frames_1s(self):
        return self._nb_frames_1s
