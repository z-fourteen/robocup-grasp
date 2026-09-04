# VS Code Codex 远程连接故障诊断

日期：2026-09-04

> 本文已脱敏：主机路径、内部地址、端口和凭据位置均使用泛化描述。原始诊断记录不应提交到公开仓库。

## 1. 结论

当前 VS Code Codex 插件不能正常使用，主要是两个独立问题叠加：

1. 当前 SSH 窗口实际使用的是旧版 VS Code Server `1.104.3`，而新版 Codex 扩展需要 `chatSessionsProvider` 提案 API，导致扩展激活时报错。
2. 远程扩展宿主继承了失效的本地代理地址，但远程主机当前没有对应监听服务，导致 Codex 无法访问 ChatGPT、GitHub 和相关接口。

终端中的 Codex 能使用，是因为终端使用了独立的 Codex CLI 和自定义 OpenAI 兼容 provider；VS Code 插件还需要 ChatGPT 官方服务和账户认证，两者不是同一条认证/网络路径。

## 2. 证据

### 2.1 远程 Server 版本不匹配

当前 SSH 会话中的命令版本为：

```text
VS Code 1.104.3
```

当前扩展宿主日志多次报告其他扩展不兼容 `Code 1.104.3`，说明该窗口确实运行在旧 Server 上。

Codex 日志中的报错为：

```text
Extension 'openai.chatgpt' CANNOT use API proposal: chatSessionsProvider
```

远程主机上已经存在新版 Server（版本 `1.135.0`）：

```text
VS Code Server 1.135.0
```

对应的历史 Codex 会话可以完成 `Initialize received`，没有上述 `chatSessionsProvider` 错误。因此问题不是新版扩展包缺少该声明，而是当前窗口加载了旧 Server/旧扩展组合。

### 2.2 远程代理地址失效

Codex 生成的远程 shell 快照中包含：

```text
HTTP_PROXY=http://127.0.0.1:<proxy-port>
HTTPS_PROXY=http://127.0.0.1:<proxy-port>
```

扩展宿主日志中记录：

```text
Failed to establish a socket connection to proxies: PROXY 127.0.0.1:<proxy-port>
```

Codex 日志还显示：

```text
Failed to connect to 127.0.0.1 port <proxy-port>
TypeError: fetch failed
status=432
```

远程主机的代理配置曾使用该地址，但当前检查没有发现监听。因此这是旧代理地址残留，可能来自本地 VS Code 的 `remote.SSH.remoteEnv`、系统环境，或在代理运行时启动的长期 VS Code 进程。

### 2.3 终端和插件使用的 Codex 不同

终端使用：

```text
codex-cli 0.147.0
```

终端配置指向自定义 provider：

```toml
base_url = "http://<custom-provider>/v1"
wire_api = "responses"
```

插件使用的是扩展自带的 Codex：

```text
codex-cli 0.153.0
```

扩展文档明确以 ChatGPT 账户登录为使用方式。插件日志还提示：

```text
chatgpt authentication required for remote plugin catalog; api key auth is not supported
```

所以仅在 `config.toml` 中配置自定义 API，不能保证 VS Code 插件的 ChatGPT 账户、插件目录和官方服务功能可用。

## 3. 恢复步骤

### 3.1 先切换到新版远程 Server

在本地 VS Code 客户端执行：

1. 更新 VS Code 客户端，确保版本足够新，建议至少使用与远程 Server `1.135.0` 匹配的版本。
2. 关闭所有连接到该远程主机的 VS Code 窗口。
3. 打开命令面板，执行 `Remote-SSH: Kill VS Code Server on Host...`，选择对应主机。
4. 重新连接 Remote-SSH；必要时执行 `Developer: Reload Window`。

重新连接后，在远程终端验证：

```bash
code --version
command -v code
```

预期版本为 `1.135.0`，路径应指向 `Stable-08d4889.../server/bin/remote-cli/code`。然后检查 Codex 日志，不应再出现 `CANNOT use API proposal: chatSessionsProvider`。

### 3.2 清理或修复代理

如果远程主机不需要代理：

- 在本地 VS Code 设置中检查并删除 `remote.SSH.remoteEnv` 中的 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 等设置。
- 检查本地 VS Code 启动环境和 Remote-SSH 配置，清除指向本地代理端口的旧环境变量。
- 完成上述修改后必须完全重连远程窗口，已有扩展宿主不会自动更新环境变量。

如果确实需要代理：

- 确保代理服务运行在远程主机配置的本地监听端口；或者
- 配置远程主机可访问的代理地址，并同时配置监听地址、防火墙和认证。

本地电脑的 `127.0.0.1` 对远程主机不是同一个回环地址，不能直接填写为远程代理。

### 3.3 认证方式

在 VS Code Codex 面板中使用 ChatGPT 账户登录。若只需要自定义 OpenAI 兼容 API，继续使用终端 Codex CLI 更合适；插件是否支持该 provider 不能从当前扩展设置中配置，插件本身没有公开的 provider/base URL 设置项。

## 4. 安全事项

Codex 认证文件当前权限过宽，且包含明文 API key。建议：

1. 立即撤销并重新生成该 API key。
2. 将文件权限改为仅用户可读写：

```bash
chmod 600 ~/.codex/auth.json
```

不要把现有 key 复制到聊天、日志或截图中。

## 5. 诊断范围

本次检查为只读诊断，没有修改项目文件、VS Code Server 或远程连接状态。项目源码工作区没有相关改动。
