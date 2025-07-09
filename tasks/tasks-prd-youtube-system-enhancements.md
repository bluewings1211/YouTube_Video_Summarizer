# YouTube 系統增強功能 - 任務列表

## Relevant Files

- `src/api/batch.py` - 批次處理 API 端點和路由處理
- `src/api/batch.test.py` - 批次處理 API 的單元測試
- `src/api/history.py` - 歷史紀錄刪除和重新處理 API（擴展現有功能）
- `src/api/history.test.py` - 歷史紀錄 API 的單元測試
- `src/api/notifications.py` - 通知和 webhook API 端點
- `src/api/notifications.test.py` - 通知 API 的單元測試
- `src/services/batch_service.py` - 批次處理業務邏輯服務
- `src/services/batch_service.test.py` - 批次處理服務的單元測試
- `src/services/queue_service.py` - 佇列管理服務
- `src/services/queue_service.test.py` - 佇列服務的單元測試
- `src/services/notification_service.py` - 通知和 webhook 服務
- `src/services/notification_service.test.py` - 通知服務的單元測試
- `src/services/history_service.py` - 擴展現有歷史服務以支持刪除功能
- `src/services/semantic_analysis_service.py` - 語意分析和時間戳改進服務
- `src/services/semantic_analysis_service.test.py` - 語意分析服務的單元測試
- `src/database/models.py` - 擴展現有模型以支持批次處理和狀態追蹤
- `src/database/batch_models.py` - 批次處理相關的資料庫模型
- `src/database/notification_models.py` - 通知設定相關的資料庫模型
- `src/refactored_nodes/batch_processing_nodes.py` - 批次處理流程節點
- `src/refactored_nodes/semantic_analysis_nodes.py` - 語意分析流程節點
- `src/utils/webhook_client.py` - Webhook 客戶端工具
- `src/utils/webhook_client.test.py` - Webhook 客戶端的單元測試
- `src/utils/semantic_analyzer.py` - 語意分析和分組工具
- `src/utils/semantic_analyzer.test.py` - 語意分析工具的單元測試
- `alembic/versions/xxx_add_batch_processing.py` - 批次處理資料庫遷移
- `alembic/versions/xxx_add_notifications.py` - 通知系統資料庫遷移
- `alembic/versions/xxx_add_semantic_analysis.py` - 語意分析資料庫遷移

### Notes

- 單元測試應該與對應的代碼文件放在同一目錄
- 使用 `pytest` 來運行測試：`pytest [可選的測試文件路徑]`
- 資料庫遷移使用 Alembic 工具管理
- 新功能應該整合到現有的 PocketFlow 工作流系統中
- 所有 API 端點都應該遵循現有的 FastAPI 模式

## Tasks

- [ ] 1.0 實作歷史紀錄刪除和重新處理功能
  - [ ] 1.1 擴展 HistoryService 以支持刪除特定影片記錄
  - [ ] 1.2 實作資料庫層面的級聯刪除邏輯
  - [ ] 1.3 在 history.py API 中添加刪除端點
  - [ ] 1.4 實作重新處理 API 端點
  - [ ] 1.5 添加重新處理時的快取清除機制
  - [ ] 1.6 實作刪除操作的事務管理和回滾
  - [ ] 1.7 編寫歷史紀錄刪除和重新處理的測試

- [ ] 2.0 建立批次處理和排程機制
  - [ ] 2.1 設計批次處理的資料庫模型
  - [ ] 2.2 實作 BatchService 核心業務邏輯
  - [ ] 2.3 建立批次處理的 API 端點
  - [ ] 2.4 實作佇列管理系統（QueueService）
  - [ ] 2.5 整合批次處理到 PocketFlow 工作流
  - [ ] 2.6 實作批次處理的併發控制
  - [ ] 2.7 添加批次處理的監控和日誌
  - [ ] 2.8 編寫批次處理的完整測試套件

- [ ] 3.0 實作處理狀態追蹤系統
  - [ ] 3.1 設計狀態追蹤的資料庫結構
  - [ ] 3.2 實作狀態更新機制
  - [ ] 3.3 建立狀態查詢 API 端點
  - [ ] 3.4 實作即時狀態更新邏輯
  - [ ] 3.5 整合狀態追蹤到現有處理流程
  - [ ] 3.6 實作狀態變更的事件系統
  - [ ] 3.7 添加狀態追蹤的分頁和過濾功能
  - [ ] 3.8 編寫狀態追蹤系統的測試

- [ ] 4.0 建立通知和 webhook 機制
  - [ ] 4.1 設計通知設定的資料庫模型
  - [ ] 4.2 實作 NotificationService 核心功能
  - [ ] 4.3 建立通知設定的 API 端點
  - [ ] 4.4 實作 webhook 客戶端工具
  - [ ] 4.5 整合通知到處理完成事件
  - [ ] 4.6 實作通知失敗的重試機制
  - [ ] 4.7 添加通知日誌和監控
  - [ ] 4.8 編寫通知系統的測試

- [ ] 5.0 改進 timestamped_segments 語意分析
  - [ ] 5.1 實作語意分析服務（SemanticAnalysisService）
  - [ ] 5.2 建立語意分組演算法
  - [ ] 5.3 整合 embedding 技術進行向量搜尋
  - [ ] 5.4 實作時間戳記的精確對應
  - [ ] 5.5 創建語意分析的 PocketFlow 節點
  - [ ] 5.6 整合語意分析到現有處理流程
  - [ ] 5.7 優化語意分析的效能
  - [ ] 5.8 編寫語意分析的測試