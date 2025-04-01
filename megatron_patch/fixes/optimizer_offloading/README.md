# Optimizer Offloading 修复

Patch文件仅适用于 `Megatron-LM-250328`, 对于其他版本的Megatron-LM，您需要手动修改代码

> Megatron-LM Commit ID: 6ba97dd37150a6bfba03d31808674211cf2a4d0d

## 使用Patch文件修复

```
cp Pai-Megatron-Patch/megatron_patch/fixes/optimizer_offloading/fix_optimizer_offloading.patch Pai-Megatron-Patch/Megatron-LM-250328
cd Pai-Megatron-Patch/Megatron-LM-250328
git apply fix_optimizer_offloading.patch
```

## 手动修复

### Fix 1
修改 `DistributedOptimizer._copy_model_params_to_main_params`:

#### Before
```
    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            return

        ... # other code
```

#### After

```
    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            return self.optimizer._update_fp32_param_by_new_param()

        ... # other code
```

### Fix 2

为`HybridDeivceOptimizer`增加以下方法
```
    def _update_fp32_param_by_new_param(self):
        for param, fp32_param in self.param_to_fp32_param.items():
            fp32_param.data.copy_(param)
```