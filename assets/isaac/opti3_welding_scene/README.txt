OPTi3 + 용접 토치(OPTi3_weld_torch_short) 장면

열기: Isaac Sim 5.1 에서 File > Open → 이 폴더의 posco_torch.usd
  - 폴더 구조를 바꾸지 말고 통째로 두고 열어야 한다 (내부 파일을 상대 경로로 참조함).
  - 로봇: /World/indy (Neuromeka OPTi3, 관절 joint0~joint5)
  - 토치: /World/indy/link6/tcp/OPTi3_weld_torch_short (손목 끝면, link6 기준 z=0.0477 m)
  - 토치 끝(와이어 끝) 프레임: /World/indy/link6/tcp/weld_tip

주의: 임포트된 관절 Drive 강성이 매우 낮아 그냥 Play 하면 로봇이 중력에 처진다.
      코드에서 게인을 kp=1e4, kd=1e3 (rad 기준) 정도로 올려서 쓴다.
