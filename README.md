# search-relevance

该数据集包含Home Depot网站上的许多产品和真实的客户搜索词。面临的挑战是预测所提供的搜索词和产品组合的相关性得分。为了创建基本事实标签，Home Depot已将搜索/产品对众包给了多个人工评估者。
      
相关性是介于1（不相关）到3（高度相关）之间的数字。例如，搜索“ AA电池”将被视为与一组AA尺寸的电池高度相关（相关性= 3），与无绳钻机电池相关性较小（相关性= 2），与雪铲无关（相关性= 1）。

每对均由至少三个人类评估者评估。提供的相关性评分是评分的平均值。关于评级，还有三点要了解：

在relevance_instructions.docx中提供了给评估者的特定说明。
评分者无权访问属性。
评分者可以访问产品图片，而比赛不包含图片。
您的任务是预测测试集中列出的每一对的相关性。请注意，测试集包含可见和不可见搜索词。

档案说明
train.csv-训练集，包含产品，搜索和相关性分数
test.csv-测试集，包含产品和搜索。您必须预测这些对的相关性。
product_descriptions.csv-包含每个产品的文字说明。您可以通过product_uid将此表加入培训或测试集。
attributes.csv-提供有关产品子集的扩展信息（通常代表详细的技术规格）。并非每个产品都具有属性。
sample_submission.csv-显示正确提交格式的文件
relevance_instructions.docx-提供给人类评分者的说明
资料栏位
ID  -这表示（SEARCH_TERM，product_uid）对唯一ID字段
product_uid-产品的ID
product_title-产品标题
product_description-产品的文字描述（可能包含HTML内容）
search_term-搜索查询
相关性 -给定ID的相关性评级的平均值
名称 -属性名称
value-属性的值
