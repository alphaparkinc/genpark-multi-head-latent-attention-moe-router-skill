from client import MultiHeadLatentAttentionMoeRouterClient

def main():
    client = MultiHeadLatentAttentionMoeRouterClient()
    res = client.route_sparse_moe_tokens(8192, 256, 8)
    print('MoE Routing Job: ' + res['moe_routing_id'] + ' (' + str(res['sequence_length']) + ' tokens)')
    print('Experts: ' + str(res['active_experts_dispatched']) + '/' + str(res['total_experts_pool']) + ' active | KV Cache Compression: ' + str(res['kv_cache_compression_ratio_pct']) + '%')
    print('Load Balancing Loss: ' + str(res['expert_load_balancing_loss']) + ' | Speedup: ' + str(res['throughput_acceleration_factor']) + 'x')

if __name__ == '__main__':
    main()
