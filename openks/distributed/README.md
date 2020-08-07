# openks_distributed

- 改动对比

  - 包变化`paddle.fluid.incubate.fleet.base`  $ \Longrightarrow $ `openks_distributed.base` 
    - 模块变化`fleet_base.py` $ \Longrightarrow $ `BaseDistributed.py`
      - 类变化`class Fleet `  $ \Longrightarrow $ `class BaseDistributedAlgorithm`
      - 类变化`class DistributedOptimizer` $\Longrightarrow$ `class BaseDistribuedOptimizer`
    - 模块变化`role_make.py` $ \Longrightarrow $ `RoleMaker.py`
    - 模块变化`mode.py`  $ \Longrightarrow $ `mode.py`
  - 包变化`paddle.fluid.incubate.fleet.collective` $ \Longrightarrow $ `openks_distributed.gpu`
    - 模块变化`__init__.py` $ \Longrightarrow $ `GPUDistributed.py`
      - 类变化`class Collective` $ \Longrightarrow $ `class GPUDistributedAlgorithm`
      - 类变化`class CollectiveOptimizer` $ \Longrightarrow $ `class GPUDistributedOptimizer`
  - 包变化`paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler` $ \Longrightarrow $ `openks_distributed.cpu`
    - 模块变化`__init__.py`$ \Longrightarrow $`CPUDistributed.py`
      - 类变化`class DistributedTranspiler` $ \Longrightarrow $`class CPUDistributedAlgorithm`
      - 类变化`class TranspilerOptimizer`$ \Longrightarrow $`class CPUDistributedOptimizer`

- 使用

  - collective on GPU

    ```python
    from openks_distributed import KSDistributedFactory
    from openks_distributed.base import RoleMaker
    
    algorithm = KSDistributedFactory.instantiation(flag = 1)
    algorithm.init(role)
    
    optimizer = algorithm.distributed_optimzier()
    optimizer.minimize()
    # ...
    ```

  - parameter server on CPU

    ```python
    from openks_distributed import KSDistributedFactory
    from openks_distributed.base import RoleMaker
    
    algorithm = KSDistributedFactory.instantiation(flag = 0)
    algorithm.init(role)
    
    optimizer = algorithm.distributed_optimzier()
    optimizer.minimize()
    # ...
    ```

- demo 运行

  - 替换压缩包中修改后的`collective_train.py`和`distributed_train.py`
  - 将`openks_distributed` package 放在`quick-start`目录或python `site-packages`标准第三方包安装目录
  - `./run_collective.sh && ./run_parameter_server.sh`

  或者

  - 直接在本目录运行：`python -m openks_launcher --mode cpu --worker_num 2 --server_num 2 openKS_CPU_test.py`