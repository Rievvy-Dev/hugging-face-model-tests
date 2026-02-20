# -*- coding: utf-8 -*-
"""Comparação antes/depois e detecção de bias."""
from datetime import datetime
from .io_utils import read_metrics_csv
from . import config


def detect_bias(before_metrics, after_metrics):
    """
    Detecta possíveis indícios de bias/overfitting.
    
    Retorna:
        list: Problemas detectados
    """
    issues = []
    
    for model_name in before_metrics.keys():
        before = before_metrics[model_name]
        after = after_metrics.get(model_name)
        
        if not after:
            continue
        
        bleu_delta = after["bleu"] - before["bleu"]
        bleu_pct = (bleu_delta / before["bleu"] * 100) if before["bleu"] > 0 else 0
        
        chrf_delta = after["chrf"] - before["chrf"]
        chrf_pct = (chrf_delta / before["chrf"] * 100) if before["chrf"] > 0 else 0
        
        # Detectar anomalias
        if bleu_pct > config.BLEU_OVERFITTING_THRESHOLD:
            issues.append({
                "model": model_name,
                "type": "overfitting",
                "severity": "HIGH",
                "metric": "BLEU",
                "value": bleu_pct,
                "message": f"BLEU aumentou {bleu_pct:.1f}% (>{config.BLEU_OVERFITTING_THRESHOLD}%) - possível memorização"
            })
        
        if bleu_pct < config.BLEU_DEGRADATION_THRESHOLD:
            issues.append({
                "model": model_name,
                "type": "degradation",
                "severity": "HIGH",
                "metric": "BLEU",
                "value": bleu_pct,
                "message": f"BLEU caiu {bleu_pct:.1f}% (<{config.BLEU_DEGRADATION_THRESHOLD}%) - fine-tuning prejudicou generalização"
            })
        
        # Consistência
        if bleu_pct > 5 and chrf_pct < 0:
            issues.append({
                "model": model_name,
                "type": "inconsistency",
                "severity": "MEDIUM",
                "metric": "BLEU vs chr-F",
                "value": f"bleu={bleu_pct:+.1f}% chrf={chrf_pct:+.1f}%",
                "message": f"BLEU subiu mas chr-F caiu - possível anomalia métrica"
            })
    
    return issues


def compare_and_report(before_file=config.BEFORE_METRICS_FILE,
                       after_file=config.AFTER_METRICS_FILE,
                       output_file=config.COMPARISON_REPORT):
    """
    Compara resultados antes/depois e gera relatório.
    
    Args:
        before_file: Path a métricas ANTES
        after_file: Path a métricas DEPOIS
        output_file: Path ao relatório de saída
    
    Returns:
        bool: Success
    """
    
    print(f"\n{'='*80}")
    print(f"  PASSO 6: Gerar Relatório Comparativo")
    print(f"{'='*80}\n")
    
    # Ler métricas
    before_results = read_metrics_csv(before_file)
    after_results = read_metrics_csv(after_file)
    
    if not before_results or not after_results:
        print("[ERRO] Não foi possível ler os arquivos de métricas")
        return False
    
    # Detectar problemas
    issues = detect_bias(before_results, after_results)
    
    # Gerar relatório
    report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    COMPARAÇÃO: ANTES vs DEPOIS                             ║
║                         Fine-tuning Scielo                                ║
╚════════════════════════════════════════════════════════════════════════════╝

Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Datasets:
  - Antes:  {before_file}
  - Depois: {after_file}

"""
    
    # Tabela de resultados
    report += f"\n{'='*80}\n"
    report += "RESULTADOS DETALHADOS\n"
    report += f"{'='*80}\n\n"
    
    # Header da tabela
    report += f"{'Modelo':<50} {'Métrica':<10} {'Antes':>10} {'Depois':>10} {'Delta':>10} {'%Chg':>8}\n"
    report += f"{'-'*80}\n"
    
    for model_name in sorted(before_results.keys()):
        before = before_results[model_name]
        after = after_results.get(model_name)
        
        if not after:
            report += f"{model_name:<50} {'N/A':<10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8}\n"
            continue
        
        # BLEU
        bleu_delta = after["bleu"] - before["bleu"]
        bleu_pct = (bleu_delta / before["bleu"] * 100) if before["bleu"] > 0 else 0
        
        report += f"{model_name:<50} {'BLEU':<10} {before['bleu']:>10.2f} {after['bleu']:>10.2f} {bleu_delta:>+10.2f} {bleu_pct:>+7.1f}%\n"
        
        # chr-F
        chrf_delta = after["chrf"] - before["chrf"]
        chrf_pct = (chrf_delta / before["chrf"] * 100) if before["chrf"] > 0 else 0
        
        report += f"{'':50} {'chr-F':<10} {before['chrf']:>10.2f} {after['chrf']:>10.2f} {chrf_delta:>+10.2f} {chrf_pct:>+7.1f}%\n"
        
        # COMET (se disponível)
        if before.get("comet") and after.get("comet"):
            comet_delta = after["comet"] - before["comet"]
            comet_pct = (comet_delta / before["comet"] * 100) if before["comet"] > 0 else 0
            report += f"{'':50} {'COMET':<10} {before['comet']:>10.2f} {after['comet']:>10.2f} {comet_delta:>+10.2f} {comet_pct:>+7.1f}%\n"
        
        # BERTScore (se disponível)
        if before.get("bertscore") and after.get("bertscore"):
            bert_delta = after["bertscore"] - before["bertscore"]
            bert_pct = (bert_delta / before["bertscore"] * 100) if before["bertscore"] > 0 else 0
            report += f"{'':50} {'BERTScore':<10} {before['bertscore']:>10.2f} {after['bertscore']:>10.2f} {bert_delta:>+10.2f} {bert_pct:>+7.1f}%\n"
        
        report += "\n"
    
    # Análise de problemas
    if issues:
        report += f"\n{'='*80}\n"
        report += "⚠️  ALERTAS DETECTADOS\n"
        report += f"{'='*80}\n\n"
        
        for issue in issues:
            severity_icon = "❌" if issue["severity"] == "HIGH" else "⚠️ "
            report += f"{severity_icon} {issue['model']}\n"
            report += f"   {issue['message']}\n"
            report += f"   Tipo: {issue['type']} ({issue['severity']})\n\n"
    else:
        report += f"\n{'='*80}\n"
        report += "✅ SEM ALERTAS - Fine-tuning OK\n"
        report += f"{'='*80}\n\n"
    
    # Recomendações
    report += f"\n{'='*80}\n"
    report += "RECOMENDAÇÕES\n"
    report += f"{'='*80}\n\n"
    
    report += """
✅ Sinal de sucesso:
   - BLEU aumenta 5-15% (melhoria real)
   - chr-F também aumenta (consistência)
   - Sem alertas de overfitting
   
⚠️  Sinais de alerta:
   - BLEU aumenta >20% (possível memorização)
   - BLEU cai >10% (especialização prejudica generalização)
   - BLEU e chr-F divergem (problemas métricos)

📌 Próximos passos:
   1. Revisar dados de treino para duplicatas/vazamento
   2. Avaliar em dataset externo (não Scielo) para generalização
   3. Comparar com baseline fixo (versão sem fine-tuning)
   4. Se overfitting detectado:
      - Reduzir épocas (default: 5)
      - Aumentar dropout/regularização
      - Usar dataset maior

📚 Estrutura de Dados:
   - Treino: 200k exemplos (nunca testado)
   - Validação: 20k exemplos (monitoramento interno)
   - Teste: 20k exemplos (MESMO dataset antes/depois)
"""
    
    # Salvar relatório
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Relatório salvo: {output_file}")
    
    return True
