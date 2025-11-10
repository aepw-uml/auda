svr-pw:
	auda pipe run DS-YEAR-PW ST-BASIC MD-IF MD-SVR PL-SVR SHOW \
		"--inputs=location=$(LOCATION)"

svr-predict-pw:
	auda pipe run DS-YEAR-PW ST-BASIC MD-IF MD-SVR PD-SVR PL-SVR SHOW \
		"--inputs=location=$(LOCATION);x_predict=$(X-PREDICT)"
