import torch
import torch.nn as nn


class SELDNet(nn.Module):
    """
    SELDnet 모델 (PyTorch 구현)

    구조:
        CNN 블록 (pool_size 길이만큼 반복)
            Conv2D → BatchNorm → ReLU → MaxPool → Dropout
        ↓
        Permute + Reshape: CNN 출력을 RNN 입력 형태로 변환
        ↓
        Bidirectional GRU (rnn_size 길이만큼 반복)
        ↓
        두 갈래로 분기:
            SED 헤드: FNN → sigmoid  → (batch, seq_len, nb_classes)
            DOA 헤드: FNN → tanh     → (batch, seq_len, nb_classes*2 or *3)

    Args:
        data_in       : 입력 shape (batch, 2*nb_ch, seq_len, feat_len)
        data_out      : 출력 shape 리스트 [(batch, seq_len, nb_classes), (batch, seq_len, nb_classes*3)]
        dropout_rate  : dropout 비율
        nb_cnn2d_filt : CNN 필터 수
        pool_size     : 각 CNN 블록의 주파수 축 풀링 크기 리스트 (예: [8, 8, 4])
        rnn_size      : 각 GRU 레이어의 hidden size 리스트 (예: [128, 128])
        fnn_size      : 각 FC 레이어의 크기 리스트 (예: [128])
    """

    def __init__(self, data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size, rnn_size, fnn_size):
        super().__init__()  # nn.Module.__init__() 실행

        self._dropout_rate = dropout_rate
        self._pool_size = pool_size

        # 입력 shape: (batch, 2*nb_ch, seq_len, feat_len)
        nb_ch   = data_in[-3]   # 채널 수 (2*nb_ch)
        seq_len = data_in[-2]   # 시퀀스 길이
        feat_len = data_in[-1]  # 주파수 빈 수

        # --- CNN 블록 ---
        # pool_size 개수만큼 반복, 매 블록마다 주파수 축을 pool_size[i]로 줄임
        cnn_layers = []
        in_ch = nb_ch
        for pool in pool_size:
            cnn_layers += [
                nn.Conv2d(in_ch, nb_cnn2d_filt, kernel_size=(3, 3), padding=1),  # 공간 특징 추출
                nn.BatchNorm2d(nb_cnn2d_filt),                                    # 학습 안정화
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pool)),                              # 주파수 축만 줄임 (시간 축 유지)
                nn.Dropout(dropout_rate)
            ]
            in_ch = nb_cnn2d_filt
        self.cnn = nn.Sequential(*cnn_layers)

        # CNN 통과 후 주파수 축 크기 계산 (풀링 누적 적용)
        cnn_feat_len = feat_len
        for pool in pool_size:
            cnn_feat_len //= pool

        # RNN 입력 크기 = CNN 필터 수 × 줄어든 주파수 빈 수
        rnn_input_size = nb_cnn2d_filt * cnn_feat_len

        # --- Bidirectional GRU ---
        # 각 레이어의 출력 크기 = rnn_size * 2 (양방향이라 forward + backward 합쳐짐)
        # merge_mode='mul' → 원본과 동일하게 element-wise 곱
        rnn_layers = []
        in_size = rnn_input_size
        for hidden_size in rnn_size:
            rnn_layers.append(BidirectionalGRU(in_size, hidden_size, dropout_rate))
            in_size = hidden_size  # mul 방식이므로 출력 크기 = hidden_size (양방향 곱)
        self.rnn = nn.Sequential(*rnn_layers)

        # --- SED 헤드 (음원 존재 여부) ---
        # FNN 레이어 + 최종 출력 → sigmoid
        sed_layers = []
        in_size = rnn_size[-1]
        for fnn in fnn_size:
            sed_layers += [nn.Linear(in_size, fnn), nn.Dropout(dropout_rate)]
            in_size = fnn
        sed_layers.append(nn.Linear(in_size, data_out[0][-1]))  # 출력: nb_classes
        sed_layers.append(nn.Sigmoid())
        self.sed_head = nn.Sequential(*sed_layers)

        # --- DOA 헤드 (방향각 직교 좌표) ---
        # FNN 레이어 + 최종 출력 → tanh
        doa_layers = []
        in_size = rnn_size[-1]
        for fnn in fnn_size:
            doa_layers += [nn.Linear(in_size, fnn), nn.Dropout(dropout_rate)]
            in_size = fnn
        doa_layers.append(nn.Linear(in_size, data_out[1][-1]))  # 출력: nb_classes*2 or *3
        doa_layers.append(nn.Tanh())
        self.doa_head = nn.Sequential(*doa_layers)

    def forward(self, x):
        # x: (batch, 2*nb_ch, seq_len, feat_len)

        # --- CNN ---
        x = self.cnn(x)
        # x: (batch, nb_cnn2d_filt, seq_len, cnn_feat_len)

        # Permute: (batch, nb_cnn2d_filt, seq_len, cnn_feat_len)
        #        → (batch, seq_len, nb_cnn2d_filt, cnn_feat_len)
        x = x.permute(0, 2, 1, 3)

        # Reshape: (batch, seq_len, nb_cnn2d_filt, cnn_feat_len)
        #        → (batch, seq_len, nb_cnn2d_filt * cnn_feat_len)
        batch, seq_len, _, _ = x.shape
        x = x.reshape(batch, seq_len, -1)

        # --- RNN ---
        x = self.rnn(x)
        # x: (batch, seq_len, rnn_size[-1])

        # --- 두 헤드로 분기 ---
        sed = self.sed_head(x)  # (batch, seq_len, nb_classes)
        doa = self.doa_head(x)  # (batch, seq_len, nb_classes*2 or *3)

        return sed, doa


class BidirectionalGRU(nn.Module):
    """
    Bidirectional GRU with merge_mode='mul' (element-wise 곱)
    원본 Keras 코드의 merge_mode='mul'과 동일한 동작

    forward와 backward 출력을 element-wise 곱하여 반환
    → 출력 크기: (batch, seq_len, hidden_size)  (합치지 않고 곱하므로 크기 유지)
    """

    def __init__(self, input_size, hidden_size, dropout_rate):
        super(BidirectionalGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,      # (batch, seq_len, features) 형식 사용
            bidirectional=True,    # 양방향
            dropout=dropout_rate if dropout_rate > 0 else 0
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # out: (batch, seq_len, hidden_size * 2)
        #      앞 절반: forward, 뒤 절반: backward

        forward_out  = out[:, :, :out.shape[-1] // 2]   # (batch, seq_len, hidden_size)
        backward_out = out[:, :, out.shape[-1] // 2:]   # (batch, seq_len, hidden_size)

        # merge_mode='mul': element-wise 곱
        return forward_out * backward_out  # (batch, seq_len, hidden_size)
