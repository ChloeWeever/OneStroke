import 'package:flutter/material.dart';

class RecomendPage extends StatelessWidget {
  final int id;
  final String title = "欢迎小笔划们！";
  final int authorId = 0;
  final String authorName = "\"一笔一划\"团队";
  final String imgUrl = "http://dm.ecnucpp.cn:8080/1pen/recomend/0/0.webp";
  final String content =
      "号外号外！练字评价软件“一笔一划”正式上线啦！\n在这里，你可以得到一手的练字资源，\n智能AI会全程陪伴您的练字之路。\n有超多本子、超丰富的精美字体等你来解锁！\n我们的论坛也热闹非凡！快叫上你的小伙伴一起练字打卡吧！\n号外号外！练字评价软件“一笔一划”正式上线啦！\n在这里，你可以得到一手的练字资源，\n智能AI会全程陪伴您的练字之路。\n有超多本子、超丰富的精美字体等你来解锁！\n我们的论坛也热闹非凡！快叫上你的小伙伴一起练字打卡吧！\n号外号外！练字评价软件“一笔一划”正式上线啦！\n在这里，你可以得到一手的练字资源，\n智能AI会全程陪伴您的练字之路。\n有超多本子、超丰富的精美字体等你来解锁！\n我们的论坛也热闹非凡！快叫上你的小伙伴一起练字打卡吧！\n号外号外！练字评价软件“一笔一划”正式上线啦！\n在这里，你可以得到一手的练字资源，\n智能AI会全程陪伴您的练字之路。\n有超多本子、超丰富的精美字体等你来解锁！\n我们的论坛也热闹非凡！快叫上你的小伙伴一起练字打卡吧！\n";
  final String time = "2025年1月";
  final List<Comment> CommentList = [Comment(), Comment(), Comment(),Comment(), Comment(), Comment()];
  RecomendPage({super.key, required this.id}) {
    // TODO: 通过id向后端获取推荐页信息
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('今日推荐'),
        backgroundColor: Color.fromRGBO(42, 130, 228, 0.7),
      ),
      body: SingleChildScrollView(
          child: Center(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(
              height: 20,
            ),
            OrientationBuilder(builder: (context, orientation) {
              return Container(
                width: MediaQuery.of(context).size.width * 0.97,
                height: MediaQuery.of(context).size.width * 0.5625,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12.0),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.3),
                      blurRadius: 10.0,
                      offset: Offset(0, 5),
                    ),
                  ],
                ),
                child: MainArea(
                    id, title, authorId, authorName, imgUrl, content, time),
              );
            }),
            SizedBox(
              height: 20,
            ),
            Container(
              child: SizedBox(
                width: MediaQuery.of(context).size.width * 0.97,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("评论区",
                        style: TextStyle(
                            fontSize: 35, fontWeight: FontWeight.bold)),
                    SizedBox(
                      height: MediaQuery.of(context).size.height * 0.7,
                      child: ListView.builder(
                          itemCount: CommentList.length,
                          itemBuilder: (context, index) {
                            return CommentArea(CommentList[index]);
                          }),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      )),
      backgroundColor: Color.fromRGBO(245, 245, 245, 1),
    );
  }
}

class MainArea extends StatelessWidget {
  final int id;
  final String title;
  final int authorId;
  final String authorName;
  final String imgUrl;
  final String content;
  final String time;
  const MainArea(this.id, this.title, this.authorId, this.authorName, this.imgUrl,
      this.content, this.time, {super.key});
  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
            child: Padding(
          padding: EdgeInsets.only(left: 30.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                height: 20,
              ),
              Text(
                title,
                style: TextStyle(fontSize: 35, fontWeight: FontWeight.bold),
              ),
              SizedBox(
                height: 10,
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    authorName,
                    style: TextStyle(fontSize: 20, color: Colors.grey),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      // 添加关注逻辑
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                    ),
                    child: Text('关注',
                        style: TextStyle(
                            fontSize: 20, fontWeight: FontWeight.bold)),
                  ),
                ],
              ),
              SizedBox(
                height: 20,
              ),
              Expanded(
                // 修改: 使用 Expanded 替换 SizedBox
                child: SingleChildScrollView(
                  child: Text(
                    content,
                    style: TextStyle(fontSize: 20),
                  ),
                ),
              ),
              SizedBox(
                height: 10,
              ),
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Expanded(child: Container()),
                  Text(
                    time,
                    style: TextStyle(color: Colors.grey),
                  )
                ],
              ),
              SizedBox(
                height: 20,
              ),
            ],
          ),
        )),
        Expanded(
            child: Padding(
          padding: EdgeInsets.only(right: 30.0, left: 30),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [Expanded(child: Image.network(imgUrl))],
          ),
        )),
      ],
    );
  }
}

class CommentArea extends StatelessWidget {
  final Comment comment;
  const CommentArea(this.comment, {super.key});
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        SizedBox(
          height: 5,
        ),
        Container(
            width: MediaQuery.of(context).size.width * 0.97,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12.0),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.3),
                  blurRadius: 10.0,
                  offset: Offset(0, 5),
                ),
              ],
            ),
            child: Column(
              children: [
                Padding(
                  padding: EdgeInsets.only(left: 30.0, top: 20),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Image.network(
                        comment.authorImgUrl,
                        width: 55,
                        height: 55,
                      ),
                      SizedBox(
                        width: 10,
                      ),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            comment.author,
                            style: TextStyle(
                                fontSize: 25, fontWeight: FontWeight.bold),
                          ),
                          Text(
                            comment.time,
                            style: TextStyle(fontSize: 15, color: Colors.grey),
                          ),
                        ],
                      )
                    ],
                  ),
                ),
                Padding(
                  padding: EdgeInsets.only(left: 30.0, top: 20),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(
                        width: 60,
                      ),
                      SizedBox(
                          width: MediaQuery.of(context).size.width * 0.8,
                          child: Text(
                            comment.content,
                            style: TextStyle(fontSize: 20),
                          )),
                      SizedBox(
                        width: 10,
                      ),
                    ],
                  ),
                ),
                SizedBox(
                  height: 20,
                )
              ],
            )),
        SizedBox(
          height: 5,
        ),
      ],
    );
  }
}

class Comment {
  int id = 0;
  int authorId = 0;
  String author = "书画";
  String authorImgUrl = "http://dm.ecnucpp.cn:8080/1pen/user/0/portrait.webp";
  String time = "2025-2-9";
  String content = "奶酪体新字帖";
  Comment() {
    id = 0;
    authorId = 0;
    author = "书画";
    authorImgUrl = "http://dm.ecnucpp.cn:8080/1pen/user/0/portrait.webp";
    time = "2025-2-9";
    content =
        "奶酪体新字帖jhkhkhkjhjkghjfvjgvchchchgcthycthvjvsdfafargasadjakjgkjhahjkfbabfkagbkfhgajkhfjkahfkjahskjsfhaksjfhaadfadfadfadfafafadfadfadfadfafadfadfadfadfadfadfafdafadfadfa";
  }

  Comment.withParams({
    required this.id,
    required this.authorId,
    required this.author,
    required this.authorImgUrl,
    required this.time,
    required this.content,
  });
}
