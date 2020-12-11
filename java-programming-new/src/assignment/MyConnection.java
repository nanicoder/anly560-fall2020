package assignment;

import java.sql.*;
import java.util.Properties;

public abstract class MyConnection implements Connection {
	private static MyConnection conn;

	public static void main(String[] args) {
	
			try {
				Properties properties = new Properties();
				properties.setProperty("user", "root");
				properties.setProperty("password", "");
				properties.setProperty("useSSL", "false");
				Class.forName("com.mysql.jdbc.Driver");
				conn = (MyConnection) DriverManager.getConnection("jdbc:mysql://localhost:3306/sakila", properties);
				getConn();
			}catch (Exception e) {
				System.out.println(e);
			}
		}

	public static MyConnection getConn() {
		return conn;
	}
}
