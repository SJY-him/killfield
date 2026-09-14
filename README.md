# Killfield — 本地单机版

这是 [Cichlider/killfield](https://github.com/Cichlider/killfield) 的一个分支。

原项目是一个 Tank Trouble 风格的迷宫坦克 AI 实验：**Killfield** 是实时 MPC 规划器，
**Hybrid** 是 PPO 学得的策略，**Laika** 是从原版 JS 逐行移植的脚本 AI。游戏引擎、物理、
弹道和规划器用 Rust 写成，网页加载同一份 Rust/WASM 引擎，并在浏览器本地跑 Hybrid 的
前向推理。**这些核心部分本分支一行未改。**

本分支做的是另一件事：**把它改成一个纯本地、给人玩的版本**——拆掉联网排行榜，加上鼠标
操作和武器箱，并让整套工具链能在 Windows 上跑起来。

> AI 的原理（反向弹道密度场、18 动作 MPC 推演、打分函数、PPO 的 O/A/R 定义与 16 版训练
> 血统）请看原项目的 [README](https://github.com/Cichlider/killfield) 和
> [项目报告](https://cichlider.github.io/killfield/paper/)。那些内容本分支不复述。

## 本分支改了什么

### 1. 拆掉排行榜和服务端

原项目有一套很漂亮的防作弊设计：提交的不是分数而是种子加每帧输入，由 GitHub Actions
重放验证。但本地自己玩用不上，于是全部移除——榜单页、重放编解码器、提交客户端、
Cloudflare Worker 网关、CI 验证流水线和 issue 模板。

结果是**运行时零第三方请求**，拔网线照样玩。

`src/ranked.js` 重命名为 `src/opponent.js`：那个文件名同时盖着"排行榜"和"驱动对手"两件事，
看起来像可以整个删掉，实际上删了游戏就没法玩。现在它只留驱动对手的部分。

### 2. 鼠标操作

Play 模式下一个按钮循环三态：

| 模式 | 朝向 | 油门 |
|---|---|---|
| `Mouse: off` | 键盘 | 键盘 |
| `Mouse: aim` | **光标** | 键盘 |
| `Mouse: drive` | **光标** | **光标** |

`drive` 就是纯鼠标：指哪走哪，左键开火，键盘完全不用碰。油门取自光标离车身的距离，
沿用手机轮盘那条斜坡，但**单位是格而不是像素**，所以迷宫大小不影响手感：

```
< 0.35 格   油门 0      （死区——把光标压在车上就是刹车）
0.35→1.0 格 0 → 1 线性
> 1.0 格    满速
```

死区必须明显大于车体半宽（约 0.17 格），否则光标停在车上时坦克会一直蠕动。

drive 模式下**没有倒车**：车永远转向面对光标，倒车就失去了语义；想后退就把光标放到身后。

### 3. 武器箱

挂在原版移植时留下的那个空转计时器上。原注释写得很清楚：

```rust
// ---- crates (never spawn in duel mode, but the timer still consumes RNG) ----
```

计时器每帧递减、到点重置，然后什么都不生成——道具系统是为了干净的二人决斗训练**刻意摘掉**的，
但空壳不能删，因为它每次重置都要 `rng.randrange()`，删了就改变随机流、作废原作者所有基准。

四种道具，每回合场上最多 2 个，落在格子中心（永远不在墙里），距活着的坦克 1.25 格以上：

| 道具 | 效果 |
|---|---|
| **连发** | 24 发，2 帧冷却，在途上限 5 → 12（否则和普通枪没区别） |
| **散弹** | 4 次扳机，每次 5 发弹丸，±8° 扇面 |
| **护盾** | 吃掉一次致命伤，包括你自己的跳弹 |
| **激光** | 瞬发，最多 3 次反弹，射程 14 格，3 发 |

激光在一帧内解算完毕，所以必须有瞄准线——否则等你看见光束时它已经结束了。
`laser::trace` 同时服务于开火和预览，预览用的是**正在绘制的**车身角度而不是权威角度，
这样转向过程中线也牢牢贴在炮管上。路径终点落在坦克身上时，虚线变实、变亮。

**AI 看不见这些。** `duel_obs.rs` 仍是 schema 24，箱子进不了 Hybrid 的观测；
`score.rs` 不给道具定价；规划器的沙盒显式关闭道具生成。两个 AI 只会偶然踩上箱子，
永远不会走过去拿。这条不对称就是人类唯一能占到的便宜。

激光尤其如此：Hybrid 全部 113 个威胁相关维度（10 个子弹槽、威胁紧急度、来袭计数、
9 个逐动作生存展望）**都是从子弹推导的**，而激光不产生任何子弹。在它看来开火前一刻
世界完全安全。

### 4. 确定性没有被破坏

道具的位置和种类抽自 `Game::pickup_rng`，一条独立的 mulberry32 链，从不碰 `Game::rng`。
**关掉道具时，这份引擎和改动前逐位一致。** 有测试守着：

```rust
#[test]
fn enabling_crates_does_not_disturb_the_game_rng() {
    let off = run(4242, 900, false);
    let on  = run(4242, 900, true);
    assert_eq!(off.rng.state, on.rng.state, "game RNG diverged");
    ...
}
```

生成节奏也用的是独立常量（`PICKUP_SPAWN_TIMEBASE` 等）。原版那套 350 帧基数是给
人人对战调的，而对着这些 AI 一回合约 125 帧就结束了——按老节奏箱子一次都落不了地。
但又不能直接改老常量：那个空转计时器的重置**频率**取决于它，改了就会以不同频率抽 RNG。

### 5. Windows 工具链

`build.sh` 原来写死了 macOS 的 `sed -i ''` 和 `shasum`，在 Windows 的 Git Bash 上会把
`viewer.js` 改坏。现在它探测 GNU / BSD sed 和 `sha256sum` / `shasum`，两边都能跑。

它还只给 `i18n.js` 和 `hybrid.js` 打版本戳，其余模块是裸路径 `import`。重新编译后浏览器
可能握着旧模块，报出指向源码而非缓存的错：

```
SyntaxError: The requested module './src/mouse-aim.js'
does not provide an export named 'AIM_MODE_AIM'
```

现在 `src/` 下所有模块共用一个戳（逐个算 hash 会有循环依赖：给叶子打戳会改写引用方，
引用方的 hash 又变了）。另外新增了 `viewer/serve.py`，发 `no-store` 头，让这个问题
在本地根本不会发生。

## 运行

需要 Rust 工具链和 `wasm32-unknown-unknown` target。Windows 上建议用 GNU 主机工具链，
**不需要装 Visual Studio Build Tools**——引擎 crate 零依赖、无 build script、无过程宏，
交叉编译到 wasm 连主机链接器都不会调用：

```sh
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup target add wasm32-unknown-unknown
```

编译并启动：

```sh
bash viewer/build.sh      # 约 6 秒
python viewer/serve.py    # http://127.0.0.1:8000
```

换端口：`python viewer/serve.py 8001`

## 操作

| 键 | 动作 |
|---|---|
| `↑` `W` `E` | 前进 |
| `↓` `S` `D` | 后退 |
| `←` `A` / `→` `F` | 左转 / 右转 |
| `Space` `Q` `M` | 开火 |
| `R` / `P` | 换迷宫 / 暂停 |

鼠标：移动瞄准，左键开火。手机端左侧 128 方向轮盘、右侧开火。

## Play 模式的开关

- **Instant turn** —— 取消转速限制（引擎会拒绝穿墙的姿态，所以不会瞬移进墙里）
- **Mouse** —— off / aim / drive
- **Weapon crates** —— 武器箱，**默认关闭**
- **Opponent delay** —— 对手动作延迟 0–3 帧。默认 0，也就是最高难度
- **Opening pause** —— 开局停顿 0–3 秒
- **Wheel forward region** —— 只影响手机轮盘，键鼠无效

## 调试入口

沿用原项目 `?pilot=policy` 的约定：

| 参数 | 作用 |
|---|---|
| `?weapon=laser` | 每回合直接给玩家发一把（也可 `gatling` / `shotgun` / `shield`） |
| `?pilot=policy` | 让 Hybrid 代替玩家操作，25 倍速自测 |

## 验证

```sh
cargo test --manifest-path engine/Cargo.toml     # 70 passed
node --test viewer/tests/*.test.mjs              # 3 passed
node --check viewer/viewer.js
bash viewer/build.sh
```

新增测试覆盖：道具生成节奏与拾取、`step()` 是否真的调用拾取、护盾只吃一次伤害、
弹匣耗尽回落、激光反弹与自杀规则、光束进入渲染缓冲区的偏移量、预览与实弹逐点一致、
预览不改动任何状态、鼠标油门曲线与三态切换。

## 强弱关系

原作者的基准（本分支未改动 AI，所以仍然成立）：

| 对局 | 胜率 |
|---|---|
| Hybrid → Laika | 94.1% |
| Hybrid → Killfield（512 射线） | 81.8% |
| Killfield → Laika | 90.7% |

Hybrid 是三个里最强的——它打得过那个搜索规划器。

搜索规划器反而好打，是因为它有**动作承诺窗口**：`MPC_HOLD` 让它选定的动作要保持若干帧，
中间对你的新动作没有反应。Hybrid 的训练契约是"25 Hz，每帧一次，无动作承诺、无帧跳"，
没有窗口可骗。

## License

MIT，见 [LICENSE](LICENSE)。原项目版权归 Cichlider。

`viewer/assets/brand/Green.png` 为本仓库作者所有。
