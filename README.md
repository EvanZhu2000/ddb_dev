This would be the directory for dolphindb development

1. DolphinDB installation - refer to https://docs.dolphindb.com/en/Tutorials/standalone_deployment.html 
   -> current installed in 192.168.91.124:8848 (dev env) and 192.168.91.91:8848 (research env)
2. Python connection to DolphinDB - refer to https://docs.dolphindb.com/en/pydoc/QuickStart/Demo.html

Current structure - 
1) universe.csv captures universe of symbols info, recording.dos would record actual data from ctp
2) demo_live.dos and demo_minute.dos provides example of quick strategy implementation. Can run from dolphindb executor or from python connection

The above are still quite rough at this stage, and I would provide a more modularize solution in September. Yet from above can already see that the tick-to-order latency should <1 millisecond for usual case.

All of the above can be played in 192.168.91.124:8848
