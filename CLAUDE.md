# EventCorr — Claude Code 工作規則

## Git 工作流程規則

- **絕對不可以直接在 `main` 分支上編輯或 commit。**
- 所有開發工作必須在個別的 feature branch 上進行。
- 推到遠端後，確認與 `main` 無衝突才能 merge。
- Merge 完成後，將本地與遠端的 feature branch reset 到最新的 `main`。
- 每次開始新任務前，先從最新的 `main` 開一條新分支，或確認目前 feature branch 已與 `main` 同步。
