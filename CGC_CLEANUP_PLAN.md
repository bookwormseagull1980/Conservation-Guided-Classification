# CGC 第一性原理清理计划（2026-08-03）

## 用户要求
1. CGC 中不出现 RP3（内空间组件与平坦时空核心分离）
2. 消除所有非第一性原理标注（ad-hoc/临时/猜测/占位/启发式/历史）
3. 历史修改记录删除（不留"已修改"痕迹）
4. 一切必须第一性原理、可信

## 分阶段计划

### 阶段 A：验证审计报告（先确认误报与真报）
- [ ] P0-1 Jμ 核函数：**已确认误报**（代码 Q={4/9,1/9,...} 已是 Q_f²；变量名 Q 误导 → 改名 Q_SQ）
- [ ] P0-2 g₁ 双重值：**真报**（0.666 vs 0.6083 未解决 → 统一）
- [ ] S-1/S-2 frg_flow 临时系数：**真报**（−0.5/0.5 无推导 → frg_flow_rp3 已有精确版）

### 阶段 B：RP3 组件分离
- frg_flow_rp3.py、frg_trace_density.py、self_consistent_dyson.py、frg_enhancement.py、
  gravity_feedback.py、bridge.py、schema.py（RP3 部分）、analyze_rp3_flow.py、
  output/archive_20260801_rp3path/
- → 移入 output/archive/ 或独立子包，从 CGC 核心流水线解耦

### 阶段 C：消除非第一性原理标注（逐文件）
- params.py：删 RP3 参数（L_RP3）、历史函数（channel_couplings, GRAV_SQ）、统一 g₁
- frg_flow.py：删 ad-hoc −0.5/0.5 路径（由 frg_flow_rp3 精确版取代）或标注被取代
- conservation_checker.py：PROTECTION_RULES 查表 → 确认每条有第一性原理论证
- momentum/topology_classifier.py：deferred 标注 → 要么实现独立分析，要么删除
- diagram_builder.py placeholder、qgraf_backend heuristic、dyson_schwinger 0.1、
  gravity_feedback c_T O(1)、resummation、numerical_stability 等

### 阶段 D：清理历史记录
- 所有 "superseded 2026-08-01"、"HISTORICAL"、"previous ad-hoc"、"removed (date)"、
  "v1/v2 superseded" 等注释 → 删除（不留历史痕迹）

### 阶段 E：验证
- 全部测试通过（注意 RP3 组件移出后测试需更新）
- reference_output.json 更新

## 关键决策点（需用户确认）
1. RP3 组件移到哪？（output/archive/ vs 独立子包 vs 删除）
2. frg_flow.py（平坦）是否整体删除（被 frg_flow_rp3 取代）？
3. 分类器（momentum/topology）的 deferred 独立分析：实现还是删除？
