import json
import os
from openai import OpenAI
from transformers import AutoTokenizer
import pandas as pd


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# messages = load_json("../gemini-32k-64k.json")[10]['messages']
messages_data = load_json("../gemini-32k-64k.json")
system_prompt = '''
## Role and Core Task

Your core task is to assist the user in completing a complex task.
You need to analyze and think, search for information, browse web pages, and integrate information.
Throughout the problem-solving process, you will act multiple times. Each of your actions includes reasoning and tool calls. 
Follow the process and steps below throughout the task completion process.

## Task Execution Process

1.  **Initial Understanding and Information Gathering:** At the beginning of the task, it is necessary to fully understand the topic and the user's task. Collect information on the concepts and information within it using the provided tools to clarify the direction for solving the task.
2.  **In-depth Exploration:** Subsequently, delve deeper and conduct detailed information gathering and research on each sub-problem or topic.
3.  **Information Integration and Summary:** Integrate information related to the task to ensure its accuracy and reliability. When you believe that the collected information is sufficiently comprehensive and you are confident, generate a `summary` tool call.

## Guiding Principles

1.  **Comprehensiveness and Completeness:** Exhaustively search and access all content and pages related to the task to ensure comprehensive collection of information without omissions, covering all predefined sub-topics and dimensions in the user's task. The ultimately collected information should be detailed and in-depth.
2.  **User Interaction:**
    * The user will only speak once to ask a question, and thereafter your every action will be to output tool call instructions and receive execution results.
    * Unless there is a special reason, you should use the same language as the user for reasoning and tool calls. If the user has additional language instructions, or you think using another language is more effective, you can ignore this requirement.
3.  **Tool Calls:**
    * Check the tool call records in the historical conversation to avoid generating exactly the same tool call as in the history (same tool name + exactly the same parameters).
4.  **Number of Actions:**
    * To solve the problem efficiently and comprehensively, you can perform up to 8 actions if necessary, until you are sure that enough information has been collected. Please reasonably arrange each action opportunity.
5.  **Source Filtering:**
    * Filter content and information that is highly authoritative and of high quality; ignore unreliable information.
    * Filter content and information that is highly timely; discard content that you judge to be outdated and do not adopt it. 
    * If information and conclusions from different web pages differ regarding the same fact, you need to cross-verify and infer which web page should be adopted. At the same time, pay attention to choosing information that is highly timely and highly authoritative.
6. **Language:**
   * If the user's language is English, your reasoning, thinking, and tool usage should all be in English.
   * If the user's language is not English, your reasoning and thinking should be in the same language as the user's, but tool usage can be in any language, preferably English if higher-quality information is available in that language.
   * If there are additional language instructions in the user's task, the language used for your tool calls should follow the user’s requirements.

Current Time is 2025-05-10 Saturday 17:38:08 CST
'''
data = {
    "messages": [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "## 收集数学领域10大最重要的公式及其应用信息\n- [ ] 搜索数学史上被广泛认可的最重要公式列表\n- [ ] 收集每个公式的背景、发现者及历史意义\n- [ ] 研究各公式在现代科学和技术中的应用",
            "reasoning_content": None,
            "refusal": None,
            "reasoning": None,
            "tool_calls": None,
            "tool_call_id": None,
            "agent_name": None
        },
        {
            "role": "assistant",
            "content": "# 研究计划：数学领域10大最重要的公式及其应用\n\n我将系统地收集数学领域最重要的公式及其应用信息。首先需要确定哪些公式被广泛认为是最重要的，然后深入研究每个公式的背景、发现者、历史意义以及现代应用。\n\n## 初步搜索策略\n\n我将首先进行广泛搜索，确定数学史上公认的最重要公式列表，然后再深入研究每个公式的详细信息。",
            "reasoning_content": "",
            "refusal": None,
            "reasoning": None,
            "tool_calls": [
                {
                    "id": "toolu_bdrk_014mxnvKAXTk2hQhiBZS14Ue",
                    "index": 0,
                    "type": "function",
                    "function": {
                        "name": "parallel_web_search",
                        "arguments": "{\"search_terms\": [\"数学史上最重要的10大公式\",\"数学史上最具影响力的公式排名\",\"数学史中最伟大的公式及其应用\",\"数学基本公式及其在科学技术中的应用\",\"最著名数学公式及其发现者历史背景\"]}"
                    }
                }
            ],
            "tool_call_id": None,
            "agent_name": None
        },
        {
            "role": "tool",
            "content": "[\n  {\n    \"search_query\": \"数学史上最重要的10大公式\",\n    \"search_results\": \"搜索结果：{\\n  \\\"organic\\\": [\\n    {\\n      \\\"title\\\": \\\"世界上最伟大的十大公式，你都知道哪些？\\\",\\n      \\\"link\\\": \\\"https://zhuanlan.zhihu.com/p/111407308\\\",\\n      \\\"snippet\\\": \\\"英国科学期刊《物理世界》曾让读者投票评选了“最伟大的公式”，最终榜上有名的十个公式既有无人不知的1+1=2，又有著名的E=mc^2；既有简单的圆周公式，又有 ...\\\",\\n      \\\"date\\\": \\\"Mar 6, 2020\\\",\\n      \\\"position\\\": 1\\n    },\\n    {\\n      \\\"title\\\": \\\"世界上最伟大的十大公式，质能方程仅排第五，1+1=2也在其中- 知乎\\\",\\n      \\\"link\\\": \\\"https://zhuanlan.zhihu.com/p/638294938\\\",\\n      \\\"snippet\\\": \\\"英国科学期刊《物理世界》曾让读者投票评选了“最伟大的公式”，最终榜上有名的十个公式既有无人不知的1+1=2，又有著名的E=mc^2；既有简单的圆周公式，又有复杂 ...\\\",\\n      \\\"date\\\": \\\"Jun 19, 2023\\\",\\n      \\\"position\\\": 2\\n    },\\n    {\\n      \\\"title\\\": \\\"数学史上最美十大公式，第一名是谁？ - 360Doc\\\",\\n      \\\"link\\\": \\\"http://www.360doc.com/content/25/0203/13/71134344_1145828579.shtml\\\",\\n      \\\"snippet\\\": \\\"古希腊大数学家阿基米德求出圆周率的下界和上界分别为223/71和22/7，并取它们的平均值3.141851为圆周率的近似值。\\\\n\\\\n公元480年左右，中国南北朝时期的数学家祖冲计算出精确到小数点后7位的结果，给出不足近似值3.1415926和过剩近似值3.1415927。\\\\n\\\\n2024年3月15日，据美国趣味科学网站报道，在国际圆周率日，总部位于美国加州的计算机存储公司Solidigm发布声明称，该公司已将圆周率Pi（π）计算到小数点后\\\\n\\\\n约105万亿位，打破此前100万亿位的世界纪录。\\\\n\\\\n上榜理由：\\\\n\\\\n这个简洁的恒等式完美地将数学中最重要的五个基本常数 统一到一个公式中，仿佛一句诗，道尽了数学之美和自然之美，令人叹为观止！\\\\n\\\\n广义圆面积公式\\\\n\\\\n上榜理由：此公式统一了平面几何学所有图形的面积计算方法，是一种万能的图形面积计算公式，同时开创了相对几何的创立，揭示了圆周率广义概念定义奥秘，同时开创性的发现并建立了几何最小周长的数学新历史，这使0点的几何定义可以有了明确的概念，重新定义了周长的本质结构（最大周长与最小周长），同时揭开了不同图形之间的相似原理，明确了几何图形面积，周长和广义圆周率（常数率）之间的关系，此公式可以说是相对几何的创立标志，相对几何创立将成为数学史发展的新纪元。\\\",\\n      \\\"date\\\": \\\"Jan 15, 2025\\\",\\n      \\\"position\\\": 3\\n    },\\n    {\\n      \\\"title\\\": \\\"1+1=2 世界上最伟大的十大公式你知道几个？ - CAE软件\\\",\\n      \\\"link\\\": \\\"http://www.feaworks.org/about/xinwenzixun/2021-03-24-693.html\\\",\\n      \\\"snippet\\\": \\\"当前位置： 主页 \\\\\\\\> 关于我们 \\\\\\\\> 技术资讯 \\\\\\\\>\\\\n\\\\n英国科学期刊《物理世界》曾让读者投票评选了“最伟大的公式”，最终榜上有名的十个公式既有无人不知的1+1=2，又有著名的E=mc^2；既有简单的-圆周公式，又有复杂的欧拉公式……\\\\n\\\\n这些公式不仅仅是数学家和物理学家的智慧结晶，更是人类文明的集中体现。每一个公式都深深影响了人类社会的变革，甚至塑造了人类的思想。这些公式中有些你会很熟悉，有些你也许不那么熟悉。作为人类的我们有必要了解这些公式，了解人类的思想历程。\\\\n\\\\n对于电子工程师来说，傅里叶变换、麦克斯韦方程都是最应该了解和掌握的。今天我们从网络收集整理了这10著名公式，分享给大家：\\\\n\\\\n**No.10 圆的周长公式(The Length of the Circumference of a Circle)**\\\\n\\\",\\n      \\\"date\\\": \\\"Mar 24, 2021\\\",\\n      \\\"position\\\": 4\\n    },\\n    {\\n      \\\"title\\\": \\\"最著名的十大公式- ZJQCation - 博客园\\\",\\n      \\\"link\\\": \\\"https://www.cnblogs.com/vacation/p/5183385.html\\\",\\n      \\\"snippet\\\": \\\"最著名的十大公式No.10 圆的周长公式（The Length of the Circumference of a Circle） No.9 傅立叶变换（The Fourier Transform） No.8 德布罗意方程 ...\\\",\\n      \\\"date\\\": \\\"Feb 5, 2016\\\",\\n      \\\"position\\\": 5\\n    }\\n  ]\\n}\"\n  },\n  {\n    \"search_query\": \"数学史上最具影响力的公式排名\",\n    \"search_results\": \"搜索结果：{\\n  \\\"organic\\\": [\\n    {\\n      \\\"title\\\": \\\"涨知识：最伟大的10个公式，质能方程仅排第五，1 1=2也在其中\\\",\\n      \\\"link\\\": \\\"http://www.360doc.com/content/21/1221/22/57798620_1009754440.shtml\\\",\\n      \\\"snippet\\\": \\\"排名第十的是圆的周长公式，，相信大家对这个公式很熟悉，因为在小学就会学到这个公式，同时这个公式的适用性很强，圆形被很多人认为是世界上最完美的图形，天上的太阳和月球是球形的，就连地球本身也接近于球形，或许我们的宇宙的几何形状也是趋于球形的也说不定。\\\\n\\\\n圆的周长公式很好地展现了圆的曲线美，还展示出了一个神奇的数字“ **π**”，也就是圆的周长与直径的比值，π在人类的历史中有着很重要的意义，目前π已经计算到了小数点后62.8万亿位，从理论上来说π包含了所有数字组合，如果用数字代替字母的话，圆周率中就包含了宇宙的一切秘密……\\\\n\\\\n**9.傅立叶变换公式**\\\\n\\\\n排名第九的公式是 **傅里叶变换公式**，这是一个相对复杂的公式，在日常生活中几乎接触不到，不过对于很多需要学习这个公式的大学生来说，这个公式绝对会让他们十分的头疼。\\\\n\\\\n在不同的研究领域，傅里叶变换具有多种不同的变体，最初傅里叶变换是作为热过程的解析分析的工具出现的，而在后来这个公式的适用性越来越广泛，简单来说这个公式是一种 **分析信号的方法，可以分析信号的成分，也可用这些成分合成信号**。\\\\n\\\\n我们可以把傅里叶变换视为一个 **将一个域内的函数转变为另一个域内函数的工** 具，这个公式可以帮助我们更充分地了解这个宇宙。\\\\n\\\\n**8.德布罗意方程组**\\\\n\\\\n在高中时期学习“ **波粒二象性**”的时候，应该会简单了解一下这个方程组，德布罗意认为宇宙中的物质都具备波粒二象性，也就是所有的粒子都具备波的性质，波粒二象性是粒子的基本性质之一，这样就可以解释很多难题了。\\\",\\n      \\\"date\\\": \\\"Dec 21, 2021\\\",\\n      \\\"position\\\": 1\\n    },\\n    {\\n      \\\"title\\\": \\\"世界上最伟大的十大公式，质能方程仅排第五，1+1=2也在其中- 知乎\\\",\\n      \\\"link\\\": \\\"https://zhuanlan.zhihu.com/p/638294938\\\",\\n      \\\"snippet\\\": \\\"大一统理论逐渐演变成了“ **统一四种基本力的理论**”，包括霍金在内的大部分物理学家都认为未来人类会找到这样的理论，目前人类物理学面临的最大问题就是相对论和量子力学之间的矛盾，相对论完美的解释了引力，而量子力学统一了另外三种基本力，而大一统理论就是统一了四种基本力的理论。\\\\n\\\\n目前，最有希望成为大一统理论的是“ **M理论**”也就是我们常说的“弦理论”，这个理论不仅仅统一了四种基本力，还从另外一个角度解释了宇宙和宇宙之外的事情。如果弦理论可以被证实，或许会成为人类最伟大的理论和公式。\\\\n\\\\n**那么在科学家的眼中，人类历史上排名前十的公式分别是什么呢？**\\\\n\\\\n英国科学期刊《物理世界》曾让读者投票评选了“最伟大的公式”，最终榜上有名的十个公式既有无人不知的1+1=2，又有著名的E=mc^2；既有简单的圆周公式，又有复杂的欧拉公式……\\\\n\\\\n这些公式不仅仅是数学家和物理学家的智慧结晶，更是人类文明的集中体现。每一个公式都深深影响了人类社会的变革，甚至塑造了人类的思想。这些公式中有些你会很熟悉，有些你也许不那么熟悉。作为人类的我们有必要了解这些公式，了解人类的思想历程。\\\\n\\\\n今天收集整理了这10著名公式，分享给大家。\\\\n\\\\n**最伟大的十个公式**\\\\n\\\\nNo.10 圆的周长公式(The Length of the Circumference of a Circle)创立者：古人意义：自然界之美的数学表达。\\\",\\n      \\\"date\\\": \\\"Jun 19, 2023\\\",\\n      \\\"position\\\": 2\\n    },\\n    {\\n      \\\"title\\\": \\\"世界上最伟大的十大公式\\\",\\n      \\\"link\\\": \\\"https://zhuanlan.zhihu.com/p/437709796\\\",\\n      \\\"snippet\\\": \\\"英国科学期刊《物理世界》曾让读者投票评选了“最伟大的公式”，最终榜上有名的十个公式既有无人不知的1+1=2，又有著名的E=mc^2；既有简单的圆周公式， ...\\\",\\n      \\\"date\\\": \\\"Nov 25, 2021\\\",\\n      \\\"attributes\\\": {\\n        \\\"Missing\\\": \\\"排名 | Show results with:排名\\\"\\n      },\\n      \\\"position\\\": 3\\n    },\\n    {\\n      \\\"title\\\": \\\"牛顿、欧拉等十大伟大科学公式揭晓，爱因斯坦质能方程只排第六 - 搜狐\\\",\\n      \\\"link\\\": \\\"https://www.sohu.com/a/856700168_121798711\\\",\\n      \\\"snippet\\\": \\\"在科学的发展历程中，公式与方程不仅仅是工具，更是人类理解宇宙、探索自然万象的桥梁。近日，英国权威科学期刊《物理世界》发起的“历史上最伟大的公式”评选结果出炉，这十个公式涵盖了从古代到现代物理、数学的高峰，展示了科学家们的智慧与创造力。\\\\n\\\\n评选的第十位是德布罗意方程组，由路易·维克多·德布罗意在1924年提出。他在这套方程中首次提出了物质波的概念，指出所有物质不仅具有粒子性，同时也具有波动性。这一理论为量子力学的发展奠定了扎实的基础，德布罗意在1929年因此获得诺贝尔物理学奖。\\\\n\\\\n第九位是最小作用量原理，该原理最早由皮埃尔·莫佩尔蒂于1744至1746年之间提出，强调在力学系统中，系统的运动状态可以通过作用量最小化来获取。这个原理的提出为经典力学的发展提供了重要的理论支持。\\\\n\\\\n在第八位上，简洁而精彩的“1+1=2”公式，无需过多解释，但它是数学基础中的基石，代表了数学逻辑的严谨性。\\\\n\\\\n第七位被玻尔兹曼公式占据，玻尔兹曼在1872年提出了这一熵公式，该公式对封闭体系的无序程度进行了量化，揭示了特定过程中的不可逆性。尽管他在生前此理论遭遇了不少争议，但如今其影响力早已超越了他所生活的时代。\\\\n\\\\n而第六位，由阿尔伯特·爱因斯坦于1905年提出的质能方程也在榜上，但它的排名却让许多人感到意外。\\\",\\n      \\\"date\\\": \\\"Feb 5, 2025\\\",\\n      \\\"position\\\": 4\\n    },\\n    {\\n      \\\"title\\\": \\\"最伟大的10个公式转载 - CSDN博客\\\",\\n      \\\"link\\\": \\\"https://blog.csdn.net/chehec2010/article/details/80570148\\\",\\n      \\\"snippet\\\": \\\"在2015年，罗切斯特大学的科学家们在氢原子能级的量子力学计算中发现了圆周率相同的公式，是世界上最伟大的十个公式之一。\\\",\\n      \\\"date\\\": \\\"Jun 4, 2018\\\",\\n      \\\"attributes\\\": {\\n        \\\"Missing\\\": \\\"影响 排名\\\"\\n      },\\n      \\\"position\\\": 5\\n    }\\n  ]\\n}\"\n  },\n  {\n    \"search_query\": \"数学史中最伟大的公式及其应用\",\n    \"search_results\": \"搜索结果：{\\n  \\\"organic\\\": [\\n    {\\n      \\\"title\\\": \\\"世界上最伟大的十个公式及其实际应用（三） - 360Doc\\\",\\n      \\\"link\\\": \\\"http://www.360doc.com/content/22/0111/02/75591611_1012744715.shtml\\\",\\n      \\\"snippet\\\": \\\"发文工具\\\\n\\\\n撰写\\\\n\\\\n网文摘手\\\\n\\\\n文档\\\\n\\\\n视频\\\\n\\\\n思维导图\\\\n\\\\n随笔\\\\n\\\\n相册\\\\n\\\\n原创同步助手\\\\n\\\\n其他工具\\\\n\\\\n图片转文字\\\\n\\\\n文件清理\\\\n\\\\nAI助手\\\\n\\\\n留言交流\\\\n\\\\n本来不想写这一篇了，因为阅读量让我很伤心。相比数学和科普，大家对奇闻异事更感兴趣。但是既然开始了，我就要善始善终。今天我就介绍一下剩余的5个公式，不期待有多少人看，内容就简略些吧，跟喜欢我前两篇的朋友说声抱歉。\\\\n\\\\n道歉\\\\n\\\\n6.欧拉公式。提出者：莱昂哈德·欧拉。\\\\n\\\\n欧拉公式\\\\n\\\\n它是世界公认的最完美的数学公式之一，提出者是数学领域的大神级天才人物——欧拉。他生于1707年4月15日，瑞士人，13岁考入巴塞尔大学，15岁大学毕业，16岁获得硕士学位。他是数学史上最多产的数学家，平均每年写出八百多页的论文，还写了大量的力学、分析学、几何学、变分法等的课本，《无穷小分析引论》、《微分学原理》、《积分学原理》等都是他的经典著作。晚年的他因为研究数学过于劳累失去了一只眼睛的视力，但他并没有影响生活，还是一个好丈夫、好父亲，很多数学推算甚至是在他孩子的尿布上、手纸上完成的。\\\\n\\\\n欧拉\\\\n\\\\n欧拉公式的另一个形式如下图。它是用在复分析领域的公式，将三角函数与复数指数函数相关联。\\\\n\\\\n欧拉公式\\\\n\\\",\\n      \\\"position\\\": 1\\n    },\\n    {\\n      \\\"title\\\": \\\"世界上最伟大的十大公式（修改版） - 知乎专栏\\\",\\n      \\\"link\\\": \\\"https://zhuanlan.zhihu.com/p/507918922\\\",\\n      \\\"snippet\\\": \\\"因此我摘录官方评价：“薛定谔方程是世界原子物理学文献中应用最广泛、影响最大的公式。”由于对量子力学的杰出贡献，薛定谔获得1933年诺贝尔物理奖。 另外薛定谔虽然姓薛，但 ...\\\",\\n      \\\"position\\\": 2\\n    },\\n    {\\n      \\\"title\\\": \\\"五个方程式五次革命，人类科学史上最伟大的公式 - 网易\\\",\\n      \\\"link\\\": \\\"https://www.163.com/dy/article/I8TQSR9H055253ZY.html\\\",\\n      \\\"snippet\\\": \\\"英国科学期刊《物理世界》曾让读者投票评选了“最伟大的公式”，最终榜上有名的十个公式，既有无人不知的勾股定理，又有著名的质能方程；既有简单的圆周公式，又 ...\\\",\\n      \\\"date\\\": \\\"Jul 6, 2023\\\",\\n      \\\"position\\\": 3\\n    },\\n    {\\n      \\\"title\\\": \\\"最伟大的10个公式转载 - CSDN博客\\\",\\n      \\\"link\\\": \\\"https://blog.csdn.net/chehec2010/article/details/80570148\\\",\\n      \\\"snippet\\\": \\\"在1926年，由奥地利理论物理学家薛定谔提出，用量子力学来描述对象的状态，可以这么理解薛定谔方程在量子力学中的意义与经典力学中的牛顿第二定律是一样的，是世界上最伟大的十个公式之一。\\\\n\\\\n5.质能方程\\\\n\\\\n\\\\n\\\\n提出者:爱因斯坦\\\\n\\\\n由阿尔伯特·爱因斯坦提出，在狭义相对论中，质能方程很好的阐释了质量与能量的关系，彻底颠覆人类固有的思想，促使人类文明发展，是世界上最伟大的十个公式之一。\\\\n\\\\n4.勾股定理\\\\n\\\\n\\\\n\\\\n提出者:毕达哥拉斯(也相信中国商朝的商高提出)\\\\n\\\\n勾股定理现约有500种证明方法，是数学定理中证明方法最多的定理之一，也是数学应用最广泛的定理之一，是世界上最伟大的十个公式之一。\\\\n\\\\n3.牛顿第二定律\\\\n\\\\n\\\\n\\\\n提出者:艾萨克·牛顿\\\\n\\\\n牛顿第二定律可以说是经典物理学的基本，还应用于日常生活的各个方面，很好的说明了物体运动状态的变化和对它作用的力之间的关系，是世界上最伟大的十个公式之一。\\\\n\\\\n2.欧拉公式\\\\n\\\\n\\\\n\\\\n提出者:欧拉\\\\n\\\\n现在应用的很多数学公式，都是欧拉发现的，不是一个公式，涉及数学和力学很多的分支，数学家们评价它是“上帝创造的公式”，是世界上最伟大的十个公式之一。\\\\n\\\\n1.麦克斯韦方程组\\\\n\\\\n\\\\n\\\\n提出者:麦克斯韦\\\\n\\\\n这个方程组将电场和磁场集成在一起，是一种偏微分方程，可以这么说，没有电磁场理论，现代文明很渺茫，是现代的电力科技与电子科技的基础，是世界上最伟大的十个公式之一。\\\",\\n      \\\"date\\\": \\\"Jun 4, 2018\\\",\\n      \\\"position\\\": 4\\n    },\\n    {\\n      \\\"title\\\": \\\"世界上最伟大的10个公式！这一条每个人都知道！_薛定谔 - 搜狐\\\",\\n      \\\"link\\\": \\\"https://www.sohu.com/a/295291435_438411\\\",\\n      \\\"snippet\\\": \\\"哈喽大家好，美好的周末，明Sir来给大家分享一些轻松的知识。\\\\n\\\\n英国科学期刊《 **物理世界**》曾让读者投票评选了“ **最伟大的公式**”，最终榜上有名的十个公式既有无人不知的1+1=2，又有著名的E=mc2；既有简单的圆周公式，又有复杂的欧拉公式……\\\\n\\\\n\\\\n\\\\n从什么时候起我们开始厌恶数学？可能就是在课堂上，只要稍微失神一下就可以完全听不懂老师说的是什么开始吧...\\\\n\\\\n不过数学是一种很奇妙的东西，这个地球上有多少伟大的智慧曾耗尽一生，才最终写下一个等号。每当你解不开方程的时候，不妨换一个角度想，会发现特别有意思哦~\\\\n\\\\n**下面这十个伟大的公式，你都知道吗？**\\\\n\\\\n**圆的周长公式（The Length of the Circumference of a Circle）**\\\\n\\\",\\n      \\\"date\\\": \\\"Feb 17, 2019\\\",\\n      \\\"position\\\": 5\\n    }\\n  ]\\n}\"\n  },\n  {\n    \"search_query\": \"数学基本公式及其在科学技术中的应用\",\n    \"search_results\": \"搜索结果：{\\n  \\\"organic\\\": [\\n    {\\n      \\\"title\\\": \\\"[PDF] 科学研究中的数学、数学方法 - CORE\\\",\\n      \\\"link\\\": \\\"https://core.ac.uk/download/41452736.pdf\\\",\\n      \\\"snippet\\\": \\\"当用数学公式来表示一个定理、定律时，就可 以运用数学知识进行计算和合乎逻辑的推导，便于我们更好 地掌握规律，有些还可以得到一些科学的预测。\\\",\\n      \\\"position\\\": 1\\n    },\\n    {\\n      \\\"title\\\": \\\"应用数学对中国科技未来的发展举足轻重 - 清华大学\\\",\\n      \\\"link\\\": \\\"https://www.tsinghua.edu.cn/info/2013/78194.htm\\\",\\n      \\\"snippet\\\": \\\"应用数学方法只是应用数学的一部分，而应用数学的主体则是建立科学概念、构造数学模型和公式，以及发展数学理论，并作科学上的预测。 ... 基础数学）——它的目的是研究数学原理 ...\\\",\\n      \\\"position\\\": 2\\n    },\\n    {\\n      \\\"title\\\": \\\"计算机科学需要掌握的数学公式原创 - CSDN博客\\\",\\n      \\\"link\\\": \\\"https://blog.csdn.net/qq_40213824/article/details/122588554\\\",\\n      \\\"snippet\\\": \\\"**指数：**\\\\n\\\\n\\\\n\\\\n**对数：**\\\\n\\\\n在计算机科学中，除非有特别的申明，所有的对数都是以2为底的。\\\\n\\\\n\\\\n\\\\n定理1.1\\\\n\\\\n\\\\n\\\\n定理1.2\\\\n\\\\n\\\\n\\\\n其他一些公式可以以此类推：\\\\n\\\\n\\\\n\\\\n级数：\\\\n\\\\n立减 ¥\\\\n\\\\n博客等级\\\\n\\\\n\\\\n路在脚下\\\\\\\\*Let It Go。: 内存溢出\\\\n\\\\nchengp...: 客户端收到的消息为什么不打印？\\\\nThe client receive response of http header is :\\\\n\\\\ncookie-wang: 他会多分配内存，如果jar包多的话，服务器内存资源就不够了\\\\n\\\\n加贝2131: 为什么要指定内存大小，不指定会有什么问题呢？\\\\n\\\\n请填写红包祝福语或标题\\\\n\\\\n红包个数最小为10个\\\\n\\\\n红包金额最低5元\\\\n\\\\n抵扣说明：\\\\n\\\\n1.余额是钱包充值的虚拟货币，按照1:1的比例进行支付金额的抵扣。\\\\n\\\\n2.余额无法直接购买下载，可以购买VIP、付费专栏及课程。\\\\n\\\\n# 计算机科学需要掌握的数学公式\\\\n\\\\n### 热门文章\\\\n\\\\n### 最新评论\\\\n\\\\n### 大家在看\\\\n\\\\n### 最新文章\\\\n\\\\n### 目录\\\\n\\\\n### 目录\\\\n\\\\n### 最新文章\\\\n\\\\n### 目录\\\\n\\\\n计算机科学需要掌握的数学公式\\\\\\\\_计算机研究生专业要自己推导\\\",\\n      \\\"date\\\": \\\"Jan 19, 2022\\\",\\n      \\\"position\\\": 3\\n    },\\n    {\\n      \\\"title\\\": \\\"掌握数学公式应用：从理论到实践 - CSDN博客\\\",\\n      \\\"link\\\": \\\"https://blog.csdn.net/weixin_35364187/article/details/146368629\\\",\\n      \\\"snippet\\\": \\\"在数学的世界中，公式是表达变量之间关系的工具，它们无处不在，从基本的代数到复杂的几何和物理应用。在本篇博客中，我们将探讨如何将数学公式应用于实际 ...\\\",\\n      \\\"date\\\": \\\"Mar 18, 2025\\\",\\n      \\\"position\\\": 4\\n    },\\n    {\\n      \\\"title\\\": \\\"[PDF] 高等数学导论\\\",\\n      \\\"link\\\": \\\"http://www.astro.gsu.edu/~guo/daolun1.pdf\\\",\\n      \\\"snippet\\\": \\\"本“导论”是中国科学技术大学非数学专业通用的讲义,是在. 近50年的使用过程中,经过不断修订、充实而成的.与同类书相. 比,其广度有所拓宽,论证定理、公式逻辑严谨,编排内容 ...\\\",\\n      \\\"position\\\": 5\\n    }\\n  ]\\n}\"\n  },\n  {\n    \"search_query\": \"最著名数学公式及其发现者历史背景\",\n    \"search_results\": \"搜索结果：{\\n  \\\"organic\\\": [\\n    {\\n      \\\"title\\\": \\\"人类最美的23个数学公式 - 知乎专栏\\\",\\n      \\\"link\\\": \\\"https://zhuanlan.zhihu.com/p/578696042\\\",\\n      \\\"snippet\\\": \\\"3、戈特弗里德·莱布尼茨（ Gottfried Leibniz ,1646一1716)：德国哲学家、数学家，与牛顿先后独立发现微积分，数理逻辑奠基人。 4、刘徽（约225-295)：数学家， ...\\\",\\n      \\\"date\\\": \\\"Oct 31, 2022\\\",\\n      \\\"attributes\\\": {\\n        \\\"Missing\\\": \\\"背景 | Show results with:背景\\\"\\n      },\\n      \\\"position\\\": 1\\n    },\\n    {\\n      \\\"title\\\": \\\"我们的世界因为这17个公式而改天换地 - 哆嗒数学网\\\",\\n      \\\"link\\\": \\\"http://duodaa.com/blog/index.php/archives/605/\\\",\\n      \\\"snippet\\\": \\\"**历史：** 尽管归功于毕达哥拉斯，但是尚未确定他是第一个证明该公式的人。第一个明显的证据来自于欧几里德，也许巴比伦人比毕达哥拉斯早1000年就知道这个概念。\\\\n\\\\n**重要性：** 该公式是众多几何学的核心，将其与代数相联系，并且是三角学的基础。该公式对于精确的测绘、制图、导航是不可或缺的。\\\\n\\\\n**现代应用:** 三角原理用于确定全球定位系统导航（GPS）的相对位置。\\\\n\\\\n**对数及其恒等式**\\\\n\\\\n**含义：** 对数可以化乘为加。\\\\n\\\\n\\\\n\\\\n**历史：** 最初的概念是由苏格兰梅奇斯顿的地主约翰•纳皮尔在大量使用乘法运算时发现的，使得那些令人难以置信的繁琐和费时的计算会变得更容易和快捷。后来由亨利•布里格斯使其变得更加易于计算和使用。\\\\n\\\\n**重要性：** 对数是革命性的，它使得工程师和天文学家的计算更快更准确。随着计算机的问世，该公式似乎不再那么重要，但是对于科学家来说它仍然是不可或缺的。\\\\n\\\\n**现代应用:** 对数可以有助于我们理解放射性衰变。\\\\n\\\\n**微积分基本定理**\\\\n\\\\n\\\\n\\\\n**含义：** 可以计算出瞬时速率的变化\\\\n\\\\n**历史：** 微积分，正如我们所知在17世纪晚期由艾萨克•牛顿和戈特弗里德•莱布尼茨所发现。有关发现权的剽窃和先后有着长时间的争论。\\\",\\n      \\\"date\\\": \\\"Jul 2, 2016\\\",\\n      \\\"position\\\": 2\\n    },\\n    {\\n      \\\"title\\\": \\\"最伟大的10个公式转载 - CSDN博客\\\",\\n      \\\"link\\\": \\\"https://blog.csdn.net/chehec2010/article/details/80570148\\\",\\n      \\\"snippet\\\": \\\"在2015年，罗切斯特大学的科学家们在氢原子能级的量子力学计算中发现了圆周率相同的公式，是世界上最伟大的十个公式之一。 9.傅里叶变换. 提出者:傅里叶.\\\",\\n      \\\"date\\\": \\\"Jun 4, 2018\\\",\\n      \\\"position\\\": 3\\n    },\\n    {\\n      \\\"title\\\": \\\"数学史- 维基百科，自由的百科全书\\\",\\n      \\\"link\\\": \\\"https://zh.wikipedia.org/zh-hans/%E6%95%B0%E5%AD%A6%E5%8F%B2\\\",\\n      \\\"snippet\\\": \\\"目前最古老的数学文本是《普林顿322》（古巴比伦，约公元前1900年），《莱因德数学纸草书》（古埃及，约公元前2000年-1800年），以及《莫斯科数学纸草书》（古埃及，约公元前1890年） ...\\\",\\n      \\\"position\\\": 4\\n    },\\n    {\\n      \\\"title\\\": \\\"有史以来最杰出的26位数学家（及其主要成就），人类文明的推动者\\\",\\n      \\\"link\\\": \\\"https://www.163.com/dy/article/GRPAGNNM05328ZJ2.html\\\",\\n      \\\"snippet\\\": \\\"**尼尔斯-亨里克-阿贝尔（Niels Henrik Abel）**\\\\n\\\\n阿贝尔是挪威数学家，在很多数学领域做出了开创性的工作。他最著名的一个成果是首次完整地给出了高于四次的一般代数方程没有一般形式的代数解的证明。这个问题是他那个时代最著名的未解决问题之一，悬疑达250多年。他也是椭圆函数领域的开拓者，阿贝尔函数的发现者。尽管阿贝尔成就极高，却在生前没有得到认可，他的生活非常贫困。\\\\n\\\\n**大卫-希尔伯特（David Hilbert）**\\\\n\\\\n大卫-希尔伯特是德国数学家，19世纪和20世纪早期最具影响力的数学家之一。希尔伯特在许多领域发现并发展了广泛的基本思想，包括不变量理论、变分计算、交换代数、代数论、几何基础、算子的谱理论及其在积分方程中的应用、数学物理、以及数学的基础（特别是证明理论）。\\\\n\\\\n**费利克斯-克莱因（Felix Klein）**\\\\n\\\\n克莱因是高斯在哥廷根天文台的另一位继任者。他对代数作出了贡献，也因克莱因瓶的概念而闻名。\\\\n\\\\n**戈特弗里德-威廉-莱布尼茨（Gottfried Wilhelm Leibniz）**\\\\n\\\\n莱布尼茨是微积分的创始人之一。除了数学之外，他还擅长其他领域，包括哲学、政治、法律和历史。\\\",\\n      \\\"date\\\": \\\"Dec 21, 2021\\\",\\n      \\\"position\\\": 5\\n    }\\n  ]\\n}\"\n  }\n]",
            "reasoning_content": None,
            "refusal": None,
            "reasoning": None,
            "tool_calls": None,
            "tool_call_id": "toolu_bdrk_014mxnvKAXTk2hQhiBZS14Ue",
            "agent_name": None
        }
    ],
}
# messages = data['messages']
tools = load_json("tools_list.json")

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

model_path = os.environ.get("MODEL_PATH")
tokenizer = AutoTokenizer.from_pretrained(model_path)
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

use_stream = False
stream_options={"include_usage": True, "continuous_usage_stats": True}

index = 0
df = pd.DataFrame({},columns=['message','text', 'arguments', 'function_name'])
for item in messages_data:
    index += 1
    if index > 50:
        break
    messages = item['messages']
    stream = client.chat.completions.create(messages=messages,
                                            model=model,
                                            tools=tools,
                                            stream=use_stream,
                                            temperature = 0.6,
                                            top_p = 0.9,
                                            # top_k = 3,
                                            # stream_options=stream_options,
                                            )

    if use_stream:
        chunks = []
        function_name = ''
        arguments = ''
        text = ''
        for chunk in stream:
            print(chunk)
            # chunks.append(chunk)
            # if len(chunk.choices) == 0:
            if chunk.usage:
                print(f'prompt: {chunk.usage.prompt_tokens}, completion: {chunk.usage.completion_tokens}, total: {chunk.usage.total_tokens}')
            if len(chunk.choices) > 0 and chunk.choices[0].delta.tool_calls:
                print(chunk.choices[0].delta.tool_calls[0])
                arguments_delta = chunk.choices[0].delta.tool_calls[0].function.arguments
                func_name = chunk.choices[0].delta.tool_calls[0].function.name
                if arguments_delta is not None:
                    arguments += arguments_delta
                if func_name is not None:
                    function_name += func_name
            else:
                print(chunk)
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta)
                    text += chunk.choices[0].delta.content
        print(f'text: {text}, arguments: {arguments}, function_name: {function_name}')
        df.loc[len(df.index)] = [json.dumps(messages), text, arguments, function_name]
        text_length = len(tokenizer.tokenize(text))
        arguments_length = len(tokenizer.tokenize(arguments))
        if arguments_length == 0:
            break
        name_length = len(tokenizer.tokenize(function_name))
        print(f'text_len: {text_length}, arguments_len: {arguments_length}, name_len: {name_length}')
    else:
        print(stream)
        print(stream.choices[0])
        print(f"choices: {len(stream.choices)}")
        # output = stream.choices[0].message.content + stream.choices[0].message.tool_calls[0].function.arguments
        # completion_length = len(tokenizer.tokenize(output))
        # print("completion length: ", completion_length)

df.to_excel("test.xlsx", sheet_name='stat', index=False, header=True)
