# -*- coding:utf-8 -*-
from typing import Type, Callable, Tuple
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, Mlp


def window_partition(
        input_tensor: torch.Tensor,
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """
    将输入 tensor 进行窗口化
    Args:
        input_tensor: (b, c, h, w)
        window_size: (n, n)

    Returns: [b*(h/n)*(w/n), n*n, c]

    """
    B, C, H, W = input_tensor.shape
    # Unfold input
    windows = input_tensor.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # Permute and reshape to [B * windows_num, window_size[0]*window_size[1], channels]
    windows = windows.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0]*window_size[1])
    return windows


def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """
    反窗口化为原始 tensor
    Args:
        windows: [b*(h/n)*(w/n), n*n, c]
        original_size: (h, w)
        window_size: (n, n)

    Returns: (b, c, h, w)

    """
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    # output = windows.permute(0, 2, 1).contiguous()
    out = windows.view(B, H // window_size[0], W // window_size[1], -1, window_size[0], window_size[1])
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return out


def grid_partition(
        input_tensor: torch.Tensor,
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Grid partition function.

    Args:
        input_tensor (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)

    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
    """
    # Get size of input
    B, C, H, W = input_tensor.shape
    # Unfold input
    grid = input_tensor.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], grid_size[1], C]
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0]*grid_size[1], C).permute(0, 2, 1).contiguous()
    return grid


def grid_reverse(
        grid: torch.Tensor,
        original_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the grid partition.

    Args:
        grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height, width, and channels
    grid = grid.permute(0, 2, 1).contiguous()
    (H, W), C = original_size, grid.shape[-1]
    # Compute original batch size
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    out = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    out = out.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return out


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.

    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.

    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


def create_mask(masked_input: torch.Tensor, stack_nums: Tuple[int, int] = (32, 32)):
    """
    为 masked_input 创建一个 mask 向量
    :param masked_input:
    :param stack_nums:
    :return:
    """
    # mask = (masked_input != torch.zeros(masked_input.shape[-1]).to(masked_input.device))[..., 0]
    mask = torch.stack([masked_input for i in range(stack_nums[0])], dim=-1)
    mask = torch.stack([mask for i in range(stack_nums[1])], dim=-1)

    return mask


class RelativeSelfAttention(nn.Module):
    """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.

    Args:
        in_dims (int): Dims of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        drop (float, optional): Dropout ratio of output. Default: 0.0
        use_position_info (bool, optional): whether to use position information
    """

    def __init__(
            self,
            in_dims: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            drop: float = 0.,
            use_position_info: bool = False
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(RelativeSelfAttention, self).__init__()
        # Save parameters
        self.in_dims: int = in_dims
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]
        # Init layers
        self.qkv_mapping = nn.Linear(in_features=in_dims, out_features=3 * in_dims, bias=True)
        self.proj = nn.Linear(in_features=in_dims, out_features=in_dims, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)

        self.use_position_info = use_position_info
        if use_position_info:
            # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))
            # Get pair-wise relative position index for each token inside the window
            self.register_buffer(
                "relative_position_index", get_relative_position_index(grid_window_size[0], grid_window_size[1]))
            # Init relative positional bias
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.

        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
            self,
            input_tensor: torch.Tensor,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        """ Forward pass.
        Args:
            input_tensor (torch.Tensor): Input tensor of the shape [B_, C, N].
            mask (torch.Tensor): Mask tensor of the shape [B_, C, N].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B_, C, N].
        """
        # Get shape of input
        B_, C, N = input_tensor.shape
        # print('attn input:', input.shape)
        # Perform query key value mapping
        qkv = self.qkv_mapping(input_tensor).reshape(B_, C, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # print('qkv:', qkv.shape)
        q, k, v = qkv.unbind(0)  # 去除第0维度，并返回一个列表，列表长度是第0维上的数值
        # Scale query
        q = q * self.scale
        # print('q:', q.shape)
        # Compute attention maps
        if self.use_position_info:
            attn = q @ k.transpose(-2, -1) + self._get_relative_positional_bias()
        else:
            attn = q @ k.transpose(-2, -1)
        # print('attn:', attn.shape)
        if mask is not None:
            mask = mask[:, :, 0].unsqueeze(1)
            # print('变换的mask:', mask.shape)
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = self.softmax(attn)
        attn = self.proj_drop(attn)
        # Map value with attention maps
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B_, C, -1)
        # Perform final projection and dropout
        out = self.proj(out)
        return out


class MaxViTTransformerBlock(nn.Module):
    """ MaxViT Transformer block.

        With block partition:
        x ← x + Unblock(RelAttention(Block(LN(x))))
        x ← x + MLP(LN(x))

        With grid partition:
        x ← x + Ungrid(RelAttention(Grid(LN(x))))
        x ← x + MLP(LN(x))

        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.

    Args:
        last_dims (int): Last Dims of input tensor.
        partition_function (Callable): Partition function to be utilized (grid or window partition).
        reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        num_heads (int, optional): Number of attention heads. Default 32
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        use_position_info (bool, optional): whether to use position information
    """

    def __init__(
            self,
            last_dims: int,
            partition_function: Callable,
            reverse_function: Callable,
            grid_window_size: Tuple[int, int] = (7, 7),
            num_heads: int = 32,
            drop: float = 0.,
            drop_path: float = 0.1,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            use_position_info: bool = False
    ) -> None:
        """ Constructor method """
        super(MaxViTTransformerBlock, self).__init__()
        # Save parameters
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        # Init layers
        self.norm_1 = norm_layer(last_dims)
        # self.attention = RelativeSelfAttention(
        #     in_dims=last_dims,
        #     num_heads=num_heads,
        #     grid_window_size=grid_window_size,
        #     drop=drop,
        #     use_position_info=use_position_info
        # )
        self.mlp_1 = Mlp(
            in_features=last_dims,
            hidden_features=int(mlp_ratio * last_dims),
            act_layer=act_layer,
            drop=drop
        )
        # self.drop_path = nn.Dropout(drop_path)
        self.norm_2 = norm_layer(last_dims)
        self.mlp_2 = Mlp(
            in_features=last_dims,
            hidden_features=int(mlp_ratio * last_dims),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """ Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
            mask (torch.Tensor): Mask of the input tensor
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
        """
        # Save original shape
        B, C, H, W = input_tensor.shape
        # Perform partition
        input_partitioned = self.partition_function(input_tensor, self.grid_window_size)
        # print('view前:', input_partitioned.shape)
        # input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)
        # print('view后', input_partitioned.shape)
        if mask is not None:
            assert mask.shape == input_tensor.shape
            # mask = self.partition_function(mask, self.grid_window_size)
            # print(mask.shape)
        # Perform normalization, attention, and dropout
        # out = input_partitioned + self.drop_path(self.attention(self.norm_1(input_partitioned), mask))
        out = input_partitioned + self.mlp_1(self.norm_1(input_partitioned))
        # Perform normalization, MLP, and dropout
        # out = out + self.drop_path(self.mlp(self.norm_2(out)))
        out = out + self.mlp_2(self.norm_2(out))
        # Reverse partition
        out = self.reverse_function(out, (H, W), self.grid_window_size)
        # print('output:', output.shape)
        return out


class MaxViTBlock(nn.Module):
    """ MaxViT block composed of MBConv block, Block Attention, and Grid Attention.

    Args:
        in_dims (int): Number of output channels.
        out_dims (int): Number of output channels.
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        dims_change_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (32, 32)
        num_heads (int, optional): Number of attention heads. Default 32
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
        use_position_info (bool, optional): whether to use position information
        use_mask (bool, optional): whether to use mask
    """

    def __init__(
            self,
            in_dims: int = 1024,
            out_dims: int = 1024,
            grid_window_size: Tuple[int, int] = (7, 7),
            dims_change_size: Tuple[int, int] = (32, 32),
            num_heads: int = 32,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm,
            use_position_info: bool = False,
            use_mask: bool = True
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTBlock, self).__init__()
        # init parameters
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.dims_change_size = dims_change_size
        self.use_mask = use_mask
        # Init Block and Grid Transformer
        self.dims_change_linear = Mlp(
            in_features=in_dims,
            hidden_features=int(mlp_ratio * in_dims),
            act_layer=act_layer,
            out_features=out_dims,
            drop=drop
        )
        # self.dims_change_linear = nn.Linear(in_dims, out_dims)
        self.block_transformer = MaxViTTransformerBlock(
            last_dims=grid_window_size[0]*grid_window_size[-1],
            partition_function=window_partition,
            reverse_function=window_reverse,
            grid_window_size=grid_window_size,
            num_heads=num_heads,
            drop_path=drop_path,
            drop=drop,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer,
            use_position_info=use_position_info
        )
        # self.grid_transformer = MaxViTTransformerBlock(
        #     last_dims=grid_window_size[0]*grid_window_size[-1],
        #     partition_function=grid_partition,
        #     reverse_function=grid_reverse,
        #     grid_window_size=grid_window_size,
        #     num_heads=num_heads,
        #     drop_path=drop_path,
        #     drop=drop,
        #     mlp_ratio=mlp_ratio,
        #     act_layer=act_layer,
        #     norm_layer=norm_layer_transformer,
        #     use_position_info=use_position_info
        # )

    def forward(self, input_tensor: torch.Tensor, mask=None) -> torch.Tensor:
        """ Forward pass.
        Args:
            input_tensor (torch.Tensor): Input tensor of the shape [B, C_in, H, W]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2] (downscaling is optional)
        """
        out = self.dims_change_linear(input_tensor)
        # print('linear变换后：', out.shape)
        if self.use_mask:
            mask = create_mask(masked_input=mask, stack_nums=(self.dims_change_size[0], self.dims_change_size[1]))
            # print('new mask:', mask.shape)
        else:
            mask = None

        out = out.view(input_tensor.shape[0], input_tensor.shape[1], self.dims_change_size[0], self.dims_change_size[1])
        # print('linear_view：', out.shape)

        out = self.block_transformer(out, mask)
        # out = self.grid_transformer(self.block_transformer(out, mask), mask)
        # print("Grid_SA的特征:", out.shape)

        out = out.view(input_tensor.shape[0], input_tensor.shape[1], self.out_dims)
        return out


class MaxViTStage(nn.Module):
    """ Stage of the MaxViT.

    Args:
        depth (int): Depth of the stage.
        in_dims (int): Dims of input tensor.
        out_dims (int): Dims of output tensor.
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        dims_change_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (32, 32)
        num_heads (int, optional): Number of attention heads. Default 32
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
        use_position_info (bool, optional): whether to use position information
        use_mask (bool, optional): whether to use mask
    """

    def __init__(
            self,
            depth: int,
            in_dims: int,
            out_dims: int,
            grid_window_size: Tuple[int, int] = (7, 7),
            dims_change_size: Tuple[int, int] = (32, 32),
            num_heads: int = 32,
            drop: float = 0.1,
            drop_path: float = 0.1,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm,
            use_position_info: bool = False,
            use_mask: bool = True
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MaxViTStage, self).__init__()
        # Init blocks
        blocks = []
        for index in range(depth):
            blocks.append(
                MaxViTBlock(
                    in_dims=in_dims if index == 0 else out_dims,
                    out_dims=out_dims,
                    grid_window_size=grid_window_size,
                    dims_change_size=dims_change_size,
                    num_heads=num_heads,
                    drop_path=drop_path,
                    drop=drop,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer,
                    use_position_info=use_position_info,
                    use_mask=use_mask
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input_tensor: torch.Tensor, mask=None) -> torch.Tensor:
        """ Forward pass.
        Args:
            input_tensor (torch.Tensor): Input tensor of the shape [B, C_in, H*W].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2].
        """
        out = input_tensor
        for index, block in enumerate(self.blocks):
            out = block(out, mask)

        return out


class MaxViT(nn.Module):
    """ Implementation of the MaxViT proposed in:
        https://arxiv.org/pdf/2204.01697.pdf

    Args:
        init_dim (int, optional): Init dims of input tensor. Default 512
        depths (Tuple[int, ...], optional): Depth of each network stage. Default (2, 2, 5, 2)
        dims (Tuple[int, int], optional): The dim change of input tensor. Default (7, 7)
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        dims_change_size (List[Tuple[int, int], ...], optional): Last dim size to be utilized. Default (32, 32)
        num_heads (int, optional): Number of attention heads. Default 32
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
        use_position_info (bool, optional): whether to use position information
        use_mask (bool, optional): whether to use mask
    """

    def __init__(
            self,
            init_dim: int = 512,
            depths: Tuple[int, ...] = (1, 1),
            dims: Tuple[int, ...] = (1024, 2304),
            grid_window_size=[(8, 8), (8, 8)],
            dims_change_size=[(32, 32), (48, 48), (32, 32)],
            num_heads: int = 32,
            drop_path: float = 0.1,
            drop: float = 0.1,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm,
            use_position_info=False,
            use_mask=True
    ) -> None:
        super(MaxViT, self).__init__()
        # assert the necessary parameters
        assert len(depths) == len(dims) == len(grid_window_size) == len(dims_change_size)
        for idx, dims_size in enumerate(dims_change_size):
            assert dims_size[0] * dims_size[1] == dims[idx]
            assert dims_size[0] % grid_window_size[idx][0] == 0 and dims_size[1] % grid_window_size[idx][1] == 0

        # prepare the stage blocks
        stages = []
        for index, (depth, dim) in enumerate(zip(depths, dims)):
            stages.append(
                MaxViTStage(
                    depth=depth,
                    in_dims=init_dim if index == 0 else dims[index - 1],
                    out_dims=dim,
                    grid_window_size=grid_window_size[index],
                    dims_change_size=dims_change_size[index],
                    num_heads=num_heads,
                    drop_path=drop_path,
                    drop=drop,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer,
                    use_position_info=use_position_info,
                    use_mask=use_mask
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, input_tensor: torch.Tensor, mask=None) -> torch.Tensor:
        out = input_tensor
        for index, stage in enumerate(self.stages):
            out = stage(out, mask)

        return out


# maxvit = MaxViT(
#     init_dim=512,
#     depths=(1,1),
#     dims=(1024,512),
#     grid_window_size=[(8, 8),(4, 8)],
#     dims_change_size=[(32, 32),(16, 32)],
#     num_heads=8,
#     drop=0.1,
#     drop_path=0.1,
#     mlp_ratio=4.,
#     act_layer=nn.GELU,
#     norm_layer_transformer=nn.LayerNorm,
#     use_position_info=False,
#     use_mask=True
# ).to('cuda')
# #
# input = torch.rand(32, 450, 512)
# zero = torch.zeros(32, 20, 512)
# temp_input = torch.cat([input, zero], dim=1).to('cuda')
# print(temp_input.shape)
# mask = (temp_input != torch.zeros(temp_input.shape[-1]).to(temp_input.device))[..., 0]
# print('mask:', mask.shape)
# output = maxvit(temp_input, mask)
# print(output.shape)
# inputt = torch.rand(32, 200, 32, 32)
# win_size = (8,8)
# print('input:', inputt.shape)
# windows = grid_partition(input=inputt, grid_size=win_size)
# print(windows.shape)
# windows_r = grid_reverse(grid=windows, grid_size=win_size, original_size=inputt.shape[2:])
# print(windows_r.shape)
# print(torch.all(inputt == windows_r))
