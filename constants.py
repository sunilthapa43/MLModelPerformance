keep_features = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol',
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd Header Length', 'Bwd Header Length', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'Label']


DoS_Types = ['DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye']
Brute_Force_Types = ['FTP-Patator', 'SSH-Patator']
Web_Attack_types = ['Web Attack \x96 Brute Force', 'Web Attack \x96 XSS', 'Web Attack \x96 Sql Injection']
Others =['Heartbleed', 'Infiltration', 'Bot']


usnw_encode_features = ["proto", "state", "attack_cat", "service", "srcip", "dstip"]
iot_ton_encode_features = ["proto", "service","conn_state", "type", "src_ip", "dst_ip"]
nsl_kdd_encode_features = [0, 1, 2, 40]
column_names = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label"
]