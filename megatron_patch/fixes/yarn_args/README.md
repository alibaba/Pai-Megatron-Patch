# Yarn Max Position Embedding 修复

当前mcore使用了全局的`max_position_embeddings`覆盖Yarn的`original_max_position_embeddings`参数，可使用这个patch修复它

Patch文件仅适用于 `Megatron-LM-250328`

> Megatron-LM Commit ID: 6ba97dd37150a6bfba03d31808674211cf2a4d0d

## 使用Patch文件修复

```
cp Pai-Megatron-Patch/megatron_patch/fixes/yarn_args/fix_yarn_args.patch Pai-Megatron-Patch/Megatron-LM-250328
cd Pai-Megatron-Patch/Megatron-LM-250328
git apply fix_yarn_args.patch
```