package snakegame;
import javax.swing.*;
import javax.swing.JFrame;

public class SnakeGame extends JFrame{

    SnakeGame() {
        
        super("Snake Game");
        add(new Board());
        pack(); //設置視窗為最佳大小
 
        setLocationRelativeTo(null); //視窗座標
        setResizable(false);
    }

    public static void main(String[] args){
        new SnakeGame().setVisible(true); //允許JVM可以根據數據模型執行paint方法開始畫圖並顯示在螢幕上
    }
}